import asyncio
import json
import time
import traceback
import uuid
from typing import Dict

import bittensor as bt
from aioquic.asyncio import serve, QuicConnectionProtocol
from aioquic.h3.connection import H3Connection
from aioquic.h3.events import DataReceived, HeadersReceived
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent

from protocol import QueryZkProof, ProofOfWeightsSynapse, Competition
from bittensor.core.synapse import Synapse, TerminalInfo
from bittensor.core.settings import version_as_int
from bittensor.core.errors import (
    BlacklistedException,
    InvalidRequestNameError,
    NotVerifiedException,
    PriorityException,
    SynapseDendriteNoneException,
    SynapseException,
    SynapseParsingError,
    UnknownSynapseError,
)


class QuicServerProtocol(QuicConnectionProtocol):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.miner_session = kwargs.get("miner_session")
        self._h3 = H3Connection(self._quic)
        self._pending_requests = {}
        self._request_handlers = {
            "QueryZkProof": self.miner_session.queryZkProof,
            "ProofOfWeightsSynapse": self.miner_session.handle_pow_request,
            "Competition": self.miner_session.handleCompetitionRequest,
        }

    def quic_event_received(self, event: QuicEvent) -> None:
        for h3_event in self._h3.handle_event(event):
            self._handle_h3_event(h3_event)

    def _handle_h3_event(self, event) -> None:
        if isinstance(event, HeadersReceived):
            self._handle_request_headers(event)
        elif isinstance(event, DataReceived):
            self._handle_request_data(event)

    def _handle_request_headers(self, event: HeadersReceived) -> None:
        headers = {}
        method = None
        path = None

        for name, value in event.headers:
            name_str = name.decode()
            value_str = value.decode()

            if name_str == ":method":
                method = value_str
            elif name_str == ":path":
                path = value_str
            else:
                headers[name_str] = value_str

        self._pending_requests[event.stream_id] = {
            "method": method,
            "path": path,
            "headers": headers,
            "body": b"",
            "stream_ended": event.stream_ended,
        }

        if event.stream_ended:
            asyncio.create_task(self._process_request(event.stream_id))

    def _handle_request_data(self, event: DataReceived) -> None:
        if event.stream_id in self._pending_requests:
            self._pending_requests[event.stream_id]["body"] += event.data

            if event.stream_ended:
                asyncio.create_task(self._process_request(event.stream_id))

    async def _process_request(self, stream_id: int) -> None:
        start_time = time.time()
        synapse = None

        try:
            if stream_id not in self._pending_requests:
                await self._send_error_response(stream_id, 400, "Bad Request")
                return

            request_data = self._pending_requests[stream_id]

            synapse = await self._preprocess_request(request_data)

            await self._verify_body_integrity(request_data["body"], synapse)

            await self._check_blacklist(synapse)

            await self._verify_request(synapse)

            await self._assess_priority(synapse)

            response_synapse = await self._execute_handler(synapse)

            await self._send_success_response(stream_id, response_synapse, start_time)

        except Exception as e:
            bt.logging.error(f"Error processing QUIC request: {e}")
            traceback.print_exc()

            synapse = self._log_and_handle_error(synapse or Synapse(), e, start_time)
            await self._send_error_response_from_synapse(stream_id, synapse)

        finally:
            if stream_id in self._pending_requests:
                del self._pending_requests[stream_id]

    async def _preprocess_request(self, request_data: Dict) -> Synapse:
        path = request_data["path"]
        headers = request_data["headers"]

        try:
            request_name = path.split("/")[1] if path.startswith("/") else path
        except Exception:
            raise InvalidRequestNameError(f"Improperly formatted request path: {path}")

        if request_name == "QueryZkProof":
            synapse_class = QueryZkProof
        elif request_name == "ProofOfWeightsSynapse":
            synapse_class = ProofOfWeightsSynapse
        elif request_name == "Competition":
            synapse_class = Competition
        else:
            raise UnknownSynapseError(
                f"Synapse name '{request_name}' not found. Available: {list(self._request_handlers.keys())}"
            )

        try:
            synapse = synapse_class.from_headers(headers)
        except Exception:
            raise SynapseParsingError(
                f"Could not parse headers {headers} into synapse of type {request_name}"
            )

        synapse.name = request_name

        if synapse.axon is None:
            synapse.axon = TerminalInfo()

        synapse.axon.__dict__.update(
            {
                "version": int(version_as_int),
                "uuid": str(
                    self.miner_session.wallet.uuid
                    if hasattr(self.miner_session.wallet, "uuid")
                    else uuid.uuid1()
                ),
                "nonce": time.time_ns(),
                "status_code": 100,
                "hotkey": self.miner_session.wallet.hotkey.ss58_address,
            }
        )

        if synapse.dendrite is None:
            synapse.dendrite = TerminalInfo()

        message = f"{synapse.axon.nonce}.{synapse.dendrite.hotkey}.{synapse.axon.hotkey}.{synapse.axon.uuid}"
        synapse.axon.signature = (
            f"0x{self.miner_session.wallet.hotkey.sign(message).hex()}"
        )

        return synapse

    async def _verify_body_integrity(self, body: bytes, synapse: Synapse) -> None:
        if not body:
            return

        try:
            body_dict = json.loads(body.decode())
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in request body")

        for key, value in body_dict.items():
            if hasattr(synapse, key):
                setattr(synapse, key, value)

        parsed_body_hash = synapse.body_hash
        header_body_hash = getattr(synapse, "computed_body_hash", None)

        if header_body_hash and parsed_body_hash != header_body_hash:
            raise ValueError(
                f"Hash mismatch between header body hash {header_body_hash} and parsed body hash {parsed_body_hash}"
            )

    async def _check_blacklist(self, synapse: Synapse) -> None:
        try:
            if synapse.name == "QueryZkProof":
                blacklisted, reason = self.miner_session.proof_blacklist(synapse)
            elif synapse.name == "ProofOfWeightsSynapse":
                blacklisted, reason = self.miner_session.pow_blacklist(synapse)
            elif synapse.name == "Competition":
                blacklisted, reason = self.miner_session.competition_blacklist(synapse)
            else:
                blacklisted, reason = False, "Unknown synapse type"

            if blacklisted:
                bt.logging.trace(f"Blacklisted: {blacklisted}, {reason}")
                if synapse.axon is not None:
                    synapse.axon.status_code = 403
                raise BlacklistedException(
                    f"Forbidden. Key is blacklisted: {reason}.", synapse=synapse
                )

        except Exception as e:
            bt.logging.error(f"Error checking blacklist: {e}")
            if not isinstance(e, BlacklistedException):
                raise BlacklistedException("Blacklist check failed", synapse=synapse)
            raise

    async def _verify_request(self, synapse: Synapse) -> None:
        try:
            await self._default_verify(synapse)

        except Exception as e:
            bt.logging.trace(f"Verify exception {str(e)}")

            if synapse.axon is not None:
                synapse.axon.status_code = 401
            else:
                raise Exception("Synapse.axon object is None")

            raise NotVerifiedException(
                f"Not Verified with error: {str(e)}", synapse=synapse
            )

    async def _default_verify(self, synapse: Synapse) -> None:
        from bittensor_wallet import Keypair
        from bittensor.utils.axon_utils import (
            allowed_nonce_window_ns,
            calculate_diff_seconds,
        )

        if synapse.dendrite is not None:
            keypair = Keypair(ss58_address=synapse.dendrite.hotkey)
            # flake8: noqa: E501
            message = f"{synapse.dendrite.nonce}.{synapse.dendrite.hotkey}.{self.miner_session.wallet.hotkey.ss58_address}.{synapse.dendrite.uuid}.{synapse.computed_body_hash}"

            endpoint_key = f"{synapse.dendrite.hotkey}:{synapse.dendrite.uuid}"

            if synapse.dendrite.nonce is None:
                raise Exception("Missing Nonce")

            if not hasattr(self.miner_session, "nonces"):
                self.miner_session.nonces = {}

            V_7_2_0 = 7002000

            if (
                synapse.dendrite.version is not None
                and synapse.dendrite.version >= V_7_2_0
            ):
                current_time_ns = time.time_ns()
                allowed_window_ns = allowed_nonce_window_ns(
                    current_time_ns, synapse.timeout
                )

                if (
                    self.miner_session.nonces.get(endpoint_key) is None
                    and synapse.dendrite.nonce <= allowed_window_ns
                ):
                    diff_seconds, allowed_delta_seconds = calculate_diff_seconds(
                        current_time_ns, synapse.timeout, synapse.dendrite.nonce
                    )
                    # flake8: noqa: E501
                    raise Exception(
                        f"Nonce is too old: acceptable delta is {allowed_delta_seconds:.2f} seconds but request was {diff_seconds:.2f} seconds old"
                    )

                if (
                    self.miner_session.nonces.get(endpoint_key) is not None
                    and synapse.dendrite.nonce
                    <= self.miner_session.nonces[endpoint_key]
                ):
                    raise Exception("Nonce is too old, a newer one was last processed")
            else:
                if (
                    self.miner_session.nonces.get(endpoint_key) is not None
                    and synapse.dendrite.nonce
                    <= self.miner_session.nonces[endpoint_key]
                ):
                    raise Exception("Nonce is too old, a newer one was last processed")

            if synapse.dendrite.signature and not keypair.verify(
                message, synapse.dendrite.signature
            ):
                raise Exception(
                    f"Signature mismatch with {message} and {synapse.dendrite.signature}"
                )

            self.miner_session.nonces[endpoint_key] = synapse.dendrite.nonce
        else:
            raise SynapseDendriteNoneException(synapse=synapse)

    async def _assess_priority(self, synapse: Synapse) -> None:
        pass

    async def _execute_handler(self, synapse: Synapse) -> Synapse:
        handler = self._request_handlers.get(synapse.name)
        if not handler:
            raise UnknownSynapseError(f"No handler for synapse type: {synapse.name}")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, handler, synapse)

    def _log_and_handle_error(
        self, synapse: Synapse, exception: Exception, start_time: float
    ) -> Synapse:
        if isinstance(exception, SynapseException):
            synapse = exception.synapse or synapse
            bt.logging.trace(f"Forward handled exception: {exception}")
        else:
            bt.logging.trace(f"Forward exception: {traceback.format_exc()}")

        if synapse.axon is None:
            synapse.axon = TerminalInfo()

        error_id = str(uuid.uuid4())
        error_type = exception.__class__.__name__

        bt.logging.error(f"{error_type}#{error_id}: {exception}")

        if isinstance(exception, SynapseException):
            if isinstance(exception, PriorityException):
                status_code = 503
            elif isinstance(exception, UnknownSynapseError):
                status_code = 404
            elif isinstance(exception, BlacklistedException):
                status_code = 403
            elif isinstance(exception, NotVerifiedException):
                status_code = 401
            elif isinstance(exception, (InvalidRequestNameError, SynapseParsingError)):
                status_code = 400
            else:
                status_code = 500
            status_message = str(exception)
        else:
            status_code = 500
            status_message = f"Internal Server Error #{error_id}"

        synapse.axon.status_code = status_code
        synapse.axon.status_message = status_message
        synapse.axon.process_time = str(time.time() - start_time)

        return synapse

    async def _send_success_response(
        self, stream_id: int, synapse: Synapse, start_time: float
    ) -> None:
        try:
            if synapse.axon is None:
                synapse.axon = TerminalInfo()

            if synapse.axon.status_code is None:
                synapse.axon.status_code = 200

            if synapse.axon.status_code == 200 and not synapse.axon.status_message:
                synapse.axon.status_message = "Success"

            synapse.axon.process_time = time.time() - start_time

            response_data = synapse.model_dump()
            response_body = json.dumps(response_data).encode()

            headers_dict = synapse.to_headers()

            response_headers = [
                (b":status", b"200"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(response_body)).encode()),
            ]

            for key, value in headers_dict.items():
                response_headers.append((key.encode(), str(value).encode()))

            self._h3.send_headers(stream_id, response_headers)
            self._h3.send_data(stream_id, response_body, end_stream=True)
            self.transmit()

        except Exception as e:
            bt.logging.error(f"Error sending success response: {e}")
            await self._send_error_response(stream_id, 500, "Response encoding error")

    async def _send_error_response_from_synapse(
        self, stream_id: int, synapse: Synapse
    ) -> None:
        try:
            status_code = synapse.axon.status_code if synapse.axon else 500
            message = (
                synapse.axon.status_message if synapse.axon else "Internal Server Error"
            )

            error_data = {"error": message}
            response_body = json.dumps(error_data).encode()

            headers_dict = synapse.to_headers() if synapse else {}

            response_headers = [
                (b":status", str(status_code).encode()),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(response_body)).encode()),
            ]

            for key, value in headers_dict.items():
                response_headers.append((key.encode(), str(value).encode()))

            self._h3.send_headers(stream_id, response_headers)
            self._h3.send_data(stream_id, response_body, end_stream=True)
            self.transmit()

        except Exception as e:
            bt.logging.error(f"Error sending error response from synapse: {e}")
            await self._send_error_response(stream_id, 500, "Error response failed")

    async def _send_error_response(
        self, stream_id: int, status_code: int, message: str
    ) -> None:
        try:
            error_data = {"error": message}
            response_body = json.dumps(error_data).encode()

            response_headers = [
                (b":status", str(status_code).encode()),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(response_body)).encode()),
            ]

            self._h3.send_headers(stream_id, response_headers)
            self._h3.send_data(stream_id, response_body, end_stream=True)
            self.transmit()

        except Exception as e:
            bt.logging.error(f"Error sending basic error response: {e}")


class LightningServer:

    def __init__(self, miner_session, host: str = "0.0.0.0", port: int = 8092):
        self.miner_session = miner_session
        self.host = host
        self.port = port
        self.server = None

    def create_protocol(self, *args, **kwargs):
        kwargs["miner_session"] = self.miner_session
        return QuicServerProtocol(*args, **kwargs)

    def _create_quic_config(self) -> QuicConfiguration:
        config = QuicConfiguration(
            is_client=False,
            alpn_protocols=["h3"],
        )

        config.idle_timeout = 60.0
        config.max_stream_data = 1048576
        config.max_data = 10485760

        return config

    async def start(self) -> None:
        try:
            config = self._create_quic_config()

            bt.logging.info(f"Starting Lightning server on {self.host}:{self.port}")
            bt.logging.info(
                "Lightning server implements full bittensor security pipeline"
            )

            self.server = await serve(
                host=self.host,
                port=self.port,
                configuration=config,
                create_protocol=self.create_protocol,
            )

            bt.logging.success(f"Lightning server started on {self.host}:{self.port}")

        except Exception as e:
            bt.logging.error(f"Failed to start Lightning server: {e}")
            raise

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            bt.logging.info("Lightning server stopped")

    async def serve_forever(self) -> None:
        if self.server:
            await self.server.wait_closed()
