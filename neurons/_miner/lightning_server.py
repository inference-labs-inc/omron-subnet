import asyncio
import json
import os
import tempfile
import functools
import time
import uuid
from typing import Dict, Any
import bittensor as bt
from aioquic.asyncio import serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.asyncio.protocol import QuicConnectionProtocol
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime
import ipaddress


def generate_self_signed_cert():
    """Generate a self-signed certificate for QUIC"""
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Create certificate
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Omron"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )

    # Write to temporary files
    cert_path = os.path.join(tempfile.gettempdir(), "lightning_cert.pem")
    key_path = os.path.join(tempfile.gettempdir(), "lightning_key.pem")

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    with open(key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    bt.logging.debug(f"🔐 Generated SSL certificate: {cert_path}")
    bt.logging.debug(f"🔑 Generated SSL private key: {key_path}")

    return cert_path, key_path


class LightningMinerProtocol(QuicConnectionProtocol):
    """Pure Python QUIC protocol handler for miner"""

    def __init__(self, *args, miner_session=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.miner_session = miner_session
        self._stream_data = {}

    def quic_event_received(self, event):
        """Handle QUIC events"""
        try:
            # Handle different types of QUIC events
            from aioquic.quic.events import StreamDataReceived, StreamReset

            if isinstance(event, StreamDataReceived):
                bt.logging.debug(f"📥 Received stream data on stream {event.stream_id}")
                asyncio.create_task(
                    self.handle_stream_data(
                        event.stream_id, event.data, event.end_stream
                    )
                )
            elif isinstance(event, StreamReset):
                bt.logging.debug(f"🔄 Stream {event.stream_id} reset")
                if event.stream_id in self._stream_data:
                    del self._stream_data[event.stream_id]
            else:
                bt.logging.debug(f"📦 Received QUIC event: {type(event).__name__}")
        except Exception as e:
            bt.logging.error(f"Error handling QUIC event: {e}")

    async def handle_stream_data(self, stream_id: int, data: bytes, end_stream: bool):
        """Handle incoming stream data"""
        try:
            # Accumulate stream data
            if stream_id not in self._stream_data:
                self._stream_data[stream_id] = b""

            self._stream_data[stream_id] += data

            # Process complete message when stream ends
            if end_stream:
                complete_data = self._stream_data.pop(stream_id, b"")
                if complete_data:
                    await self.handle_synapse_data(complete_data, stream_id)
        except Exception as e:
            bt.logging.error(f"Error handling stream data: {e}")

    async def handle_synapse_data(self, data: bytes, stream_id: int):
        """Handle incoming synapse data"""
        try:
            # Parse JSON message
            message_str = data.decode("utf-8")
            message = json.loads(message_str)

            bt.logging.info(
                f"📨 Received {message.get('synapse_type', 'Unknown')} synapse on stream {stream_id}"
            )

            # Process synapse based on type
            synapse_type = message.get("synapse_type")
            synapse_data = message.get("data", {})

            bt.logging.debug(f"Raw synapse data: {synapse_data}")

            if synapse_type == "QueryZkProof":
                response = await self.handle_query_zk_proof(synapse_data)
            elif synapse_type == "ProofOfWeightsSynapse":
                response = await self.handle_pow_request(synapse_data)
            elif synapse_type == "Competition":
                response = await self.handle_competition_request(synapse_data)
            elif synapse_type == "Handshake":
                response = self.handle_handshake(synapse_data)
            else:
                response = {
                    "error": f"Unknown synapse type: {synapse_type}",
                    "success": False,
                }

            # Send response
            response_json = json.dumps(response)
            self._quic.send_stream_data(
                stream_id, response_json.encode("utf-8"), end_stream=True
            )

            bt.logging.info(
                f"✅ Sent response for {synapse_type} on stream {stream_id}"
            )

        except Exception as e:
            bt.logging.error(f"Error processing synapse: {e}")
            error_response = {"error": str(e), "success": False}
            error_json = json.dumps(error_response)
            try:
                self._quic.send_stream_data(
                    stream_id, error_json.encode("utf-8"), end_stream=True
                )
            except Exception:
                bt.logging.error("Failed to send error response")

    def handle_handshake(self, synapse_data: Dict[str, Any]) -> Dict[str, Any]:
        bt.logging.info("🤝 Handling Handshake")
        try:
            miner_hotkey = self.miner_session.wallet.hotkey.ss58_address
            signature = self.miner_session.wallet.hotkey.sign(miner_hotkey).hex()
            connection_id = str(uuid.uuid4())
            return {
                "miner_hotkey": miner_hotkey,
                "timestamp": int(time.time()),
                "signature": signature,
                "accepted": True,
                "connection_id": connection_id,
            }
        except Exception as e:
            bt.logging.error(f"Error in Handshake handler: {e}")
            return {
                "miner_hotkey": (
                    self.miner_session.wallet.hotkey.ss58_address
                    if self.miner_session
                    else ""
                ),
                "timestamp": int(time.time()),
                "signature": "",
                "accepted": False,
                "connection_id": "",
            }

    async def handle_query_zk_proof(
        self, synapse_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle QueryZkProof synapse"""
        bt.logging.info("🔧 Processing QueryZkProof")

        try:
            from protocol import QueryZkProof

            # Create synapse object
            synapse = QueryZkProof()
            for key, value in synapse_data.items():
                if hasattr(synapse, key) and key not in ["computed_body_hash", "axon"]:
                    if value == "":
                        setattr(synapse, key, None)
                    else:
                        setattr(synapse, key, value)

            # Call miner session handler
            if self.miner_session:
                result = await self.miner_session.query_zk_proof(synapse)
                return {
                    "query_output": (
                        result.query_output if hasattr(result, "query_output") else ""
                    ),
                    "success": True,
                    "processed_via": "python_quic_server",
                }
            else:
                return {
                    "query_output": "TEST_QUERY_RESPONSE",
                    "success": True,
                    "processed_via": "python_quic_server_test",
                }
        except Exception as e:
            bt.logging.error(f"Error in QueryZkProof handler: {e}")
            return {"error": str(e), "success": False}

    async def handle_pow_request(self, synapse_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ProofOfWeightsSynapse"""
        bt.logging.info("🔧 Processing ProofOfWeightsSynapse")

        try:
            from protocol import ProofOfWeightsSynapse

            synapse = ProofOfWeightsSynapse()
            for key, value in synapse_data.items():
                if hasattr(synapse, key) and key not in ["computed_body_hash", "axon"]:
                    setattr(synapse, key, value)

            if self.miner_session:
                result = await self.miner_session.pow_request(synapse)
                return {
                    "proof": result.proof if hasattr(result, "proof") else "",
                    "success": True,
                    "processed_via": "python_quic_server",
                }
            else:
                return {
                    "proof": "TEST_POW_RESPONSE",
                    "success": True,
                    "processed_via": "python_quic_server_test",
                }
        except Exception as e:
            bt.logging.error(f"Error in PoW handler: {e}")
            return {"error": str(e), "success": False}

    async def handle_competition_request(
        self, synapse_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle Competition synapse"""
        bt.logging.info("🔧 Processing Competition")

        try:
            from protocol import Competition

            synapse = Competition()
            for key, value in synapse_data.items():
                if hasattr(synapse, key) and key not in ["computed_body_hash", "axon"]:
                    setattr(synapse, key, value)

            if self.miner_session:
                result = await self.miner_session.competition_request(synapse)
                return {
                    "signals": result.signals if hasattr(result, "signals") else [],
                    "success": True,
                    "processed_via": "python_quic_server",
                }
            else:
                return {
                    "signals": [],
                    "success": True,
                    "processed_via": "python_quic_server_test",
                }
        except Exception as e:
            bt.logging.error(f"Error in Competition handler: {e}")
            return {"error": str(e), "success": False}


class LightningServer:
    """Pure Python Lightning QUIC server for miner"""

    def __init__(self, miner_session, bind_address: str = "0.0.0.0", port: int = 8091):
        self.miner_session = miner_session
        self.bind_address = bind_address
        self.port = port
        self.server = None
        self._running = False

    async def start(self):
        """Start the QUIC server"""
        try:
            bt.logging.info(
                f"🚀 Starting Python Lightning QUIC server on {self.bind_address}:{self.port}"
            )

            # Generate SSL certificates dynamically
            cert_path, key_path = generate_self_signed_cert()

            # Create QUIC configuration
            configuration = QuicConfiguration(
                alpn_protocols=["lightning-quic"],
                is_client=False,
            )

            # Load the dynamically generated certificates
            configuration.load_cert_chain(cert_path, key_path)

            # Start server using a partial to pass the miner_session
            protocol_factory = functools.partial(
                LightningMinerProtocol, miner_session=self.miner_session
            )

            self.server = await serve(
                self.bind_address,
                self.port,
                configuration=configuration,
                create_protocol=protocol_factory,
            )

            self._running = True
            bt.logging.success(
                f"✅ Lightning QUIC server started on {self.bind_address}:{self.port}"
            )

        except Exception as e:
            bt.logging.error(f"Failed to start Lightning server: {e}")
            raise

    async def serve_forever(self):
        """Run the server indefinitely"""
        if not self.server:
            await self.start()

        bt.logging.info("🔄 Lightning server running...")

        try:
            while self._running:
                await asyncio.sleep(1)
        except Exception as e:
            bt.logging.error(f"Server error: {e}")
            raise

    async def stop(self):
        """Stop the server"""
        bt.logging.info("🛑 Stopping Lightning server...")
        self._running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        bt.logging.success("✅ Lightning server stopped")


def create_lightning_server(
    miner_session, bind_address: str = "0.0.0.0", port: int = 8091
):
    """Factory function to create Lightning server"""
    return LightningServer(miner_session, bind_address, port)
