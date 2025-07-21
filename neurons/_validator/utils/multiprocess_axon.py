import asyncio
import multiprocessing
import pickle
import traceback
from dataclasses import dataclass
from typing import Dict, Any, Optional

import bittensor as bt
from aiohttp.client_exceptions import InvalidUrlClientError

from _validator.models.request_type import RequestType
from protocol import QueryZkProof, ProofOfWeightsSynapse
from deployment_layer.circuit_store import circuit_store
from _validator.utils.aioquic_transport import Lightning, query_axon_quic


@dataclass
class PicklableRequest:
    """Picklable version of Request for multiprocessing"""

    uid: int
    axon_ip: str
    axon_port: int
    axon_hotkey: str
    synapse_data: Dict[str, Any]
    synapse_type: str
    circuit_id: str
    request_type: RequestType
    request_hash: Optional[str] = None
    save: bool = False
    timeout: float = 120.0

    @classmethod
    def from_request(cls, request) -> "PicklableRequest":
        """Convert a Request object to PicklableRequest"""

        if isinstance(request.synapse, QueryZkProof):
            synapse_data = {
                "query_input": request.synapse.query_input,
                "query_output": request.synapse.query_output,
            }
            synapse_type = "QueryZkProof"
        elif isinstance(request.synapse, ProofOfWeightsSynapse):
            synapse_data = {
                "subnet_uid": request.synapse.subnet_uid,
                "verification_key_hash": request.synapse.verification_key_hash,
                "proof_system": (
                    request.synapse.proof_system.value
                    if hasattr(request.synapse.proof_system, "value")
                    else str(request.synapse.proof_system)
                ),
                "inputs": request.synapse.inputs,
                "proof": request.synapse.proof,
                "public_signals": request.synapse.public_signals,
            }
            synapse_type = "ProofOfWeightsSynapse"
        else:
            raise ValueError(f"Unknown synapse type: {type(request.synapse)}")

        return cls(
            uid=request.uid,
            axon_ip=request.axon.ip,
            axon_port=request.axon.port,
            axon_hotkey=request.axon.hotkey,
            synapse_data=synapse_data,
            synapse_type=synapse_type,
            circuit_id=request.circuit.id,
            request_type=request.request_type,
            request_hash=request.request_hash,
            save=request.save,
            timeout=(
                request.circuit.timeout
                if hasattr(request.circuit, "timeout") and request.circuit.timeout
                else 120.0
            ),
        )


@dataclass
class PicklableResponse:
    """Picklable version of Response for multiprocessing"""

    uid: int
    response_time: Optional[float]
    deserialized: Optional[Dict[str, Any]]
    result_data: Optional[Dict[str, Any]]
    success: bool
    error_message: Optional[str] = None
    circuit_id: str = ""
    request_hash: Optional[str] = None
    save: bool = False
    request_type: Optional[RequestType] = None

    def to_request_like(self, original_request):
        """Convert PicklableResponse back to a Request-like object"""
        from _validator.core.request import Request

        circuit = (
            circuit_store.get_circuit(self.circuit_id)
            if self.circuit_id
            else original_request.circuit
        )

        request_like = Request(
            uid=self.uid,
            axon=original_request.axon,
            synapse=original_request.synapse,
            circuit=circuit,
            request_type=self.request_type or original_request.request_type,
            inputs=original_request.inputs,
            request_hash=self.request_hash,
            response_time=self.response_time,
            deserialized=self.deserialized,
            save=self.save,
        )

        if self.success and self.result_data:
            mock_result = type("MockResult", (), {})()
            for key, value in self.result_data.items():
                setattr(mock_result, key, value)

            mock_result.dendrite = type("MockDendrite", (), {})()
            mock_result.dendrite.process_time = self.response_time

            request_like.result = mock_result

        return request_like


def _create_dendrite(wallet_config: Dict[str, str]) -> bt.dendrite:
    """Create dendrite in worker process"""
    try:
        wallet = bt.wallet(
            name=wallet_config["name"],
            hotkey=wallet_config["hotkey"],
            path=wallet_config.get("path", "~/.bittensor/wallets"),
        )
        return bt.dendrite(wallet=wallet)
    except Exception as e:
        bt.logging.error(f"Failed to create dendrite in worker: {e}")
        raise


def _recreate_synapse(synapse_data: Dict[str, Any], synapse_type: str):
    """Recreate synapse object from data"""
    if synapse_type == "QueryZkProof":
        return QueryZkProof(
            query_input=synapse_data.get("query_input"),
            query_output=synapse_data.get("query_output"),
        )
    elif synapse_type == "ProofOfWeightsSynapse":
        from execution_layer.circuit import ProofSystem

        proof_system_str = synapse_data.get("proof_system")
        if isinstance(proof_system_str, str):
            proof_system = (
                ProofSystem[proof_system_str]
                if proof_system_str in ProofSystem.__members__
                else ProofSystem.EZKL
            )
        else:
            proof_system = proof_system_str

        return ProofOfWeightsSynapse(
            subnet_uid=synapse_data.get("subnet_uid"),
            verification_key_hash=synapse_data.get("verification_key_hash"),
            proof_system=proof_system,
            inputs=synapse_data.get("inputs"),
            proof=synapse_data.get("proof", ""),
            public_signals=synapse_data.get("public_signals", ""),
        )
    else:
        raise ValueError(f"Unknown synapse type: {synapse_type}")


def _create_request_object(picklable_request: PicklableRequest):
    """Convert PicklableRequest back to a Request-like object for compatibility"""

    class RequestAxon:
        def __init__(self):
            self.ip = picklable_request.axon_ip
            self.port = picklable_request.axon_port
            self.hotkey = picklable_request.axon_hotkey

    class RequestCircuit:
        def __init__(self):
            self.timeout = picklable_request.timeout
            self.id = picklable_request.circuit_id

    class RequestObject:
        def __init__(self):
            self.uid = picklable_request.uid
            self.axon = RequestAxon()
            self.synapse = _recreate_synapse(
                picklable_request.synapse_data, picklable_request.synapse_type
            )
            self.circuit = RequestCircuit()
            self.request_type = picklable_request.request_type
            self.request_hash = picklable_request.request_hash
            self.save = picklable_request.save
            self.dendrite_headers = {}

    return RequestObject()


async def _query_single_axon_async(
    dendrite: bt.dendrite, picklable_request: PicklableRequest
) -> PicklableResponse:
    """Async version of axon query for worker process"""
    try:
        axon = bt.axon(
            ip=picklable_request.axon_ip,
            port=picklable_request.axon_port,
            hotkey=picklable_request.axon_hotkey,
        )

        synapse = _recreate_synapse(
            picklable_request.synapse_data, picklable_request.synapse_type
        )

        result = await dendrite.call(
            target_axon=axon,
            synapse=synapse,
            timeout=picklable_request.timeout,
            deserialize=False,
        )

        if not result:
            return PicklableResponse(
                uid=picklable_request.uid,
                response_time=picklable_request.timeout,
                deserialized=None,
                result_data=None,
                success=False,
                error_message="No result from axon",
                circuit_id=picklable_request.circuit_id,
                request_hash=picklable_request.request_hash,
                save=picklable_request.save,
                request_type=picklable_request.request_type,
            )

        response_time = (
            result.dendrite.process_time
            if result.dendrite.process_time is not None
            else picklable_request.timeout
        )

        deserialized = result.deserialize()

        result_data = {}
        if hasattr(result, "__dict__"):
            for key, value in result.__dict__.items():
                try:
                    pickle.dumps(value)
                    result_data[key] = value
                except (pickle.PicklingError, TypeError):
                    continue

        return PicklableResponse(
            uid=picklable_request.uid,
            response_time=response_time,
            deserialized=deserialized,
            result_data=result_data,
            success=True,
            circuit_id=picklable_request.circuit_id,
            request_hash=picklable_request.request_hash,
            save=picklable_request.save,
            request_type=picklable_request.request_type,
        )

    except InvalidUrlClientError:
        return PicklableResponse(
            uid=picklable_request.uid,
            response_time=picklable_request.timeout,
            deserialized=None,
            result_data=None,
            success=False,
            error_message=f"Invalid URL for UID {picklable_request.uid}",
            circuit_id=picklable_request.circuit_id,
            request_hash=picklable_request.request_hash,
            save=picklable_request.save,
            request_type=picklable_request.request_type,
        )
    except Exception as e:
        return PicklableResponse(
            uid=picklable_request.uid,
            response_time=picklable_request.timeout,
            deserialized=None,
            result_data=None,
            success=False,
            error_message=f"Failed to query axon for UID {picklable_request.uid}: {str(e)}",
            circuit_id=picklable_request.circuit_id,
            request_hash=picklable_request.request_hash,
            save=picklable_request.save,
            request_type=picklable_request.request_type,
        )


def multiprocess_axon_worker(
    picklable_request: PicklableRequest,
    wallet_config: Dict[str, str],
    use_quic: bool = False,
) -> PicklableResponse:
    """Worker function for multiprocess axon queries"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            if use_quic:
                lightning_client = Lightning(wallet_config)
                try:
                    request_object = _create_request_object(picklable_request)
                    quic_result = loop.run_until_complete(
                        query_axon_quic(lightning_client, request_object)
                    )

                    if quic_result is None:
                        return PicklableResponse(
                            uid=picklable_request.uid,
                            response_time=picklable_request.timeout,
                            deserialized=None,
                            result_data=None,
                            success=False,
                            error_message="QUIC query failed",
                            circuit_id=picklable_request.circuit_id,
                            request_hash=picklable_request.request_hash,
                            save=picklable_request.save,
                            request_type=picklable_request.request_type,
                        )

                    return PicklableResponse(
                        uid=quic_result.uid,
                        response_time=quic_result.response_time,
                        deserialized=quic_result.deserialized,
                        result_data=(
                            {
                                "status_code": getattr(
                                    quic_result.result.dendrite, "status_code", 200
                                )
                            }
                            if hasattr(quic_result, "result")
                            else {}
                        ),
                        success=True,
                        error_message=None,
                        circuit_id=picklable_request.circuit_id,
                        request_hash=picklable_request.request_hash,
                        save=picklable_request.save,
                        request_type=picklable_request.request_type,
                    )
                finally:
                    lightning_client.close()
            else:

                dendrite = _create_dendrite(wallet_config)
                return loop.run_until_complete(
                    _query_single_axon_async(dendrite, picklable_request)
                )
        finally:
            loop.close()

    except Exception as e:
        bt.logging.error(f"Error in multiprocess worker: {e}")
        traceback.print_exc()
        return PicklableResponse(
            uid=picklable_request.uid,
            response_time=picklable_request.timeout,
            deserialized=None,
            result_data=None,
            success=False,
            error_message=f"Worker error: {str(e)}",
            circuit_id=picklable_request.circuit_id,
            request_hash=picklable_request.request_hash,
            save=picklable_request.save,
            request_type=picklable_request.request_type,
        )


class MultiprocessAxonManager:
    """Manager class for multiprocess axon queries"""

    def __init__(
        self,
        wallet_config: Dict[str, str],
        max_workers: int = 16,
        use_quic: bool = False,
    ):
        self.wallet_config = wallet_config
        self.max_workers = max_workers
        self.use_quic = use_quic
        self.pool = None

    def start(self):
        """Start the multiprocess pool"""
        if self.pool is None:
            self.pool = multiprocessing.Pool(
                processes=self.max_workers,
                initializer=self._worker_init,
                initargs=(self.wallet_config, self.use_quic),
            )

    def stop(self):
        """Stop the multiprocess pool"""
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def _worker_init(self, wallet_config: Dict[str, str], use_quic: bool):
        """Initialize worker process"""
        import signal

        signal.signal(signal.SIGINT, signal.SIG_IGN)

        global _worker_wallet_config, _worker_use_quic
        _worker_wallet_config = wallet_config
        _worker_use_quic = use_quic

    async def query_axon(self, request) -> PicklableResponse:
        """Query axon using multiprocessing"""
        if not self.pool:
            raise RuntimeError("Pool not started")

        picklable_request = PicklableRequest.from_request(request)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.pool.apply(
                multiprocess_axon_worker,
                (picklable_request, self.wallet_config, self.use_quic),
            ),
        )

        return result

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


async def query_single_axon_multiprocess(
    multiprocess_manager: MultiprocessAxonManager, request
) -> Optional[object]:
    """
    Replacement for the original query_single_axon function using multiprocessing.

    Args:
        multiprocess_manager (MultiprocessAxonManager): The multiprocess manager instance.
        request: The request to send.

    Returns:
        Request | None: The request with results populated, or None if the request failed.
    """
    try:
        picklable_response = await multiprocess_manager.query_axon(request)

        if not picklable_response.success:
            if "Invalid URL" in (picklable_response.error_message or ""):
                bt.logging.warning(
                    f"Ignoring UID as axon is not a valid URL: {request.uid}. {request.axon.ip}:{request.axon.port}"
                )
            else:
                bt.logging.warning(
                    f"Failed to query axon for UID: {request.uid}. Error: {picklable_response.error_message}"
                )
            return None

        return picklable_response.to_request_like(request)

    except Exception as e:
        bt.logging.warning(f"Failed to query axon for UID: {request.uid}. Error: {e}")
        traceback.print_exc()
        return None
