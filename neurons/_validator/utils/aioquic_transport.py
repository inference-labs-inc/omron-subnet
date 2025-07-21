import asyncio
import json
import ssl
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import bittensor as bt
from aioquic.asyncio import connect
from aioquic.h3.connection import H3Connection
from aioquic.h3.events import (
    DataReceived,
    HeadersReceived,
    StreamReset,
)
from aioquic.quic.configuration import QuicConfiguration

from _validator.models.request_type import RequestType
from protocol import QueryZkProof, ProofOfWeightsSynapse


@dataclass
class QuicAxonInfo:
    """QUIC-compatible axon information"""

    ip: str
    port: int
    hotkey: str
    protocol: int = 4


@dataclass
class QuicRequest:
    """QUIC request data structure preserving bittensor semantics"""

    uid: int
    axon: QuicAxonInfo
    synapse: QueryZkProof | ProofOfWeightsSynapse
    circuit_timeout: float
    dendrite_headers: Dict[str, str]
    request_type: RequestType
    request_hash: Optional[str] = None
    save: bool = False


@dataclass
class QuicResponse:
    """QUIC response preserving bittensor response structure"""

    uid: int
    success: bool
    response_time: Optional[float]
    status_code: int
    headers: Dict[str, str]
    body: bytes
    deserialized: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class Lightning:
    """
    Lightning-fast QUIC-based client for communicating with bittensor axons.
    Preserves all bittensor authentication headers and signatures.
    """

    def __init__(self, wallet_config: Dict[str, str]):
        self.wallet_config = wallet_config
        self.wallet = None
        self._connection_cache: Dict[str, Any] = {}
        self._cache_lock = asyncio.Lock()
        self._init_wallet()

    def _init_wallet(self):
        """Initialize bittensor wallet for signing"""
        try:
            self.wallet = bt.wallet(
                name=self.wallet_config["name"],
                hotkey=self.wallet_config["hotkey"],
                path=self.wallet_config.get("path", "~/.bittensor/wallets"),
            )
        except Exception as e:
            bt.logging.error(f"Failed to initialize wallet in QUIC client: {e}")
            raise

    def _preprocess_synapse_for_request(
        self, target_axon: QuicAxonInfo, synapse: Any, timeout: float
    ) -> Any:
        """
        Preprocess synapse exactly like bittensor dendrite does.
        This ensures full compatibility with bittensor's authentication system.
        """
        import time
        from bittensor import __version_as_int__ as version_as_int

        synapse.timeout = timeout

        from bittensor.core.synapse import TerminalInfo

        synapse.dendrite = TerminalInfo(
            ip=getattr(self.wallet, "external_ip", "127.0.0.1"),
            version=version_as_int,
            nonce=time.time_ns(),
            uuid=str(getattr(self.wallet, "uuid", f"lightning-{int(time.time())}")),
            hotkey=self.wallet.hotkey.ss58_address,
        )

        synapse.axon = TerminalInfo(
            ip=target_axon.ip,
            port=target_axon.port,
            hotkey=target_axon.hotkey,
        )

        message = f"{synapse.dendrite.nonce}.{synapse.dendrite.hotkey}."
        f" {synapse.axon.hotkey}.{synapse.dendrite.uuid}.{synapse.body_hash}"
        synapse.dendrite.signature = f"0x{self.wallet.hotkey.sign(message).hex()}"

        return synapse

    def _create_bittensor_headers(self, synapse: Any) -> Dict[str, str]:
        """
        Create headers using bittensor's built-in to_headers method.
        This ensures perfect compatibility with bittensor's serialization.
        """
        base_headers = {
            "Content-Type": "application/json",
            "User-Agent": f"Bittensor/{bt.__version__}",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }

        synapse_headers = synapse.to_headers()

        base_headers.update(synapse_headers)

        return base_headers

    def _create_quic_config(
        self, verify_mode: ssl.VerifyMode = ssl.CERT_NONE
    ) -> QuicConfiguration:
        """Create QUIC configuration optimized for bittensor communication"""
        config = QuicConfiguration(
            is_client=True,
            alpn_protocols=["h3"],
            verify_mode=verify_mode,
        )

        config.idle_timeout = 30.0
        config.max_stream_data = 1048576
        config.max_data = 10485760

        return config

    async def _establish_connection(
        self, host: str, port: int
    ) -> Tuple[Any, H3Connection]:
        """Establish QUIC connection with caching"""
        connection_key = f"{host}:{port}"

        async with self._cache_lock:
            if connection_key in self._connection_cache:
                transport, h3_conn = self._connection_cache[connection_key]
                if not transport.is_closing():
                    return transport, h3_conn

        try:
            config = self._create_quic_config()

            transport, protocol = await connect(
                host,
                port,
                configuration=config,
                create_protocol=lambda: H3Connection(config),
            )

            await asyncio.sleep(0.1)

            async with self._cache_lock:
                self._connection_cache[connection_key] = (transport, protocol)

            return transport, protocol

        except Exception as e:
            bt.logging.error(
                f"Failed to establish QUIC connection to {host}:{port}: {e}"
            )
            raise

    async def _send_h3_request(
        self,
        h3_conn: H3Connection,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: bytes = b"",
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Send HTTP/3 request over QUIC"""

        h3_headers = [
            (":method", method),
            (":path", path),
            (":scheme", "https"),
            (":authority", f"{headers.get('Host', 'localhost')}"),
        ]

        for key, value in headers.items():
            if not key.startswith(":") and key.lower() != "host":
                h3_headers.append((key.lower(), value))

        stream_id = h3_conn.get_next_available_stream_id()
        h3_conn.send_headers(stream_id=stream_id, headers=h3_headers)

        if body:
            h3_conn.send_data(stream_id=stream_id, data=body, end_stream=True)
        else:
            h3_conn.end_stream(stream_id)

        response_headers = {}
        response_body = b""
        status_code = 200

        while True:
            events = h3_conn.next_event()
            if not events:
                await asyncio.sleep(0.01)
                continue

            for event in events:
                if isinstance(event, HeadersReceived) and event.stream_id == stream_id:
                    for name, value in event.headers:
                        if name == b":status":
                            status_code = int(value.decode())
                        else:
                            response_headers[name.decode()] = value.decode()

                elif isinstance(event, DataReceived) and event.stream_id == stream_id:
                    response_body += event.data
                    if event.stream_ended:
                        return status_code, response_headers, response_body

                elif isinstance(event, StreamReset) and event.stream_id == stream_id:
                    raise Exception(f"Stream reset: {event.error_code}")

            if len(response_body) > 0 and not any(
                isinstance(e, DataReceived) and e.stream_ended for e in events
            ):
                continue

            break

        return status_code, response_headers, response_body

    async def query_axon(self, request: QuicRequest) -> QuicResponse:
        """
        Query axon using QUIC while preserving bittensor authentication.
        This is the main entry point that replaces the original HTTP dendrite call.
        """
        start_time = time.time()

        try:

            preprocessed_synapse = self._preprocess_synapse_for_request(
                request.axon, request.synapse, request.circuit_timeout
            )

            headers = self._create_bittensor_headers(preprocessed_synapse)
            headers.update(request.dendrite_headers)

            endpoint = preprocessed_synapse.__class__.__name__

            body_data = preprocessed_synapse.model_dump()
            body = json.dumps(body_data).encode()
            headers["Content-Length"] = str(len(body))
            headers["Host"] = f"{request.axon.ip}:{request.axon.port}"

            transport, h3_conn = await self._establish_connection(
                request.axon.ip, request.axon.port
            )

            status_code, response_headers, response_body = await asyncio.wait_for(
                self._send_h3_request(h3_conn, "POST", f"/{endpoint}", headers, body),
                timeout=request.circuit_timeout,
            )

            response_time = time.time() - start_time

            deserialized = None
            if response_body:
                try:
                    deserialized = json.loads(response_body.decode())
                except json.JSONDecodeError as e:
                    bt.logging.warning(f"Failed to decode response JSON: {e}")

            return QuicResponse(
                uid=request.uid,
                success=200 <= status_code < 300,
                response_time=response_time,
                status_code=status_code,
                headers=response_headers,
                body=response_body,
                deserialized=deserialized,
                error_message=(
                    None if 200 <= status_code < 300 else f"HTTP {status_code}"
                ),
            )

        except asyncio.TimeoutError:
            return QuicResponse(
                uid=request.uid,
                success=False,
                response_time=request.circuit_timeout,
                status_code=408,
                headers={},
                body=b"",
                error_message="Request timeout",
            )
        except Exception as e:
            response_time = time.time() - start_time
            bt.logging.error(f"QUIC request failed for UID {request.uid}: {e}")
            return QuicResponse(
                uid=request.uid,
                success=False,
                response_time=response_time,
                status_code=500,
                headers={},
                body=b"",
                error_message=str(e),
            )

    async def _async_close(self):
        """Async cleanup of connections with proper locking"""
        async with self._cache_lock:
            for connection_key, (transport, h3_conn) in self._connection_cache.items():
                try:
                    if not transport.is_closing():
                        transport.close()
                except Exception as e:
                    bt.logging.warning(
                        f"Error closing QUIC connection {connection_key}: {e}"
                    )
            self._connection_cache.clear()

    def close(self):
        """Clean up connections - sync version for compatibility"""
        try:

            loop = asyncio.get_event_loop()
            if loop.is_running():

                asyncio.create_task(self._async_close())
            else:

                loop.run_until_complete(self._async_close())
        except RuntimeError:

            try:
                for connection_key, (transport, h3_conn) in list(
                    self._connection_cache.items()
                ):
                    try:
                        if not transport.is_closing():
                            transport.close()
                    except Exception as e:
                        bt.logging.warning(
                            f"Error closing QUIC connection {connection_key}: {e}"
                        )
                self._connection_cache.clear()
            except Exception as e:
                bt.logging.warning(f"Error during sync cleanup: {e}")

    def __del__(self):
        """Cleanup on deletion"""
        self.close()


async def query_axon_quic(lightning_client: Lightning, request) -> Optional[object]:
    """
    Drop-in replacement for query_single_axon using QUIC transport.
    Preserves all bittensor signatures and headers while using QUIC for transport.
    """
    try:

        quic_axon = QuicAxonInfo(
            ip=request.axon.ip,
            port=request.axon.port,
            hotkey=request.axon.hotkey,
            protocol=4,
        )

        dendrite_headers = {}
        if hasattr(request, "dendrite_headers"):
            dendrite_headers = request.dendrite_headers

        quic_request = QuicRequest(
            uid=request.uid,
            axon=quic_axon,
            synapse=request.synapse,
            circuit_timeout=request.circuit.timeout,
            dendrite_headers=dendrite_headers,
            request_type=request.request_type,
            request_hash=request.request_hash,
            save=request.save,
        )

        quic_response = await lightning_client.query_axon(quic_request)

        if not quic_response.success:
            if "Invalid URL" in (quic_response.error_message or ""):
                bt.logging.warning(
                    f"Ignoring UID as axon is not reachable via QUIC: {request.uid}."
                    f" {request.axon.ip}:{request.axon.port}"
                )
            else:
                bt.logging.warning(
                    f"Failed to query axon via QUIC for UID: {request.uid}. "
                    f"Error: {quic_response.error_message}"
                )
            return None

        class BittensorResult:
            def __init__(self):
                self.dendrite = BittensorDendrite()

            def deserialize(self):
                return quic_response.deserialized

        class BittensorDendrite:
            def __init__(self):
                self.process_time = quic_response.response_time
                self.status_code = quic_response.status_code
                self.status_message = (
                    "Success" if quic_response.success else quic_response.error_message
                )

                for key, value in quic_response.headers.items():
                    setattr(self, key.replace("-", "_"), value)

        request.result = BittensorResult()
        request.response_time = quic_response.response_time
        request.deserialized = quic_response.deserialized

        return request

    except Exception as e:
        bt.logging.warning(
            f"Failed to query axon via QUIC for UID: {request.uid}. Error: {e}"
        )
        traceback.print_exc()
        return None
