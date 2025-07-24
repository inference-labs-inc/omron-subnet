import asyncio
import json
from typing import Dict, Any
import bittensor as bt
from aioquic.asyncio import serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.asyncio.protocol import QuicConnectionProtocol


class LightningMinerProtocol(QuicConnectionProtocol):
    """Pure Python QUIC protocol handler for miner"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.miner_session = kwargs.get("miner_session")

    def quic_event_received(self, event):
        """Handle QUIC events"""
        try:
            if hasattr(event, "data"):
                # Received synapse data
                asyncio.create_task(self.handle_synapse_data(event.data))
        except Exception as e:
            bt.logging.error(f"Error handling QUIC event: {e}")

    async def handle_synapse_data(self, data: bytes):
        """Handle incoming synapse data"""
        try:
            # Parse JSON message
            message_str = data.decode("utf-8")
            message = json.loads(message_str)

            bt.logging.info(
                f"📨 Received {message.get('synapse_type', 'Unknown')} synapse"
            )

            # Process synapse based on type
            synapse_type = message.get("synapse_type")
            synapse_data = message.get("data", {})

            if synapse_type == "QueryZkProof":
                response = await self.handle_query_zk_proof(synapse_data)
            elif synapse_type == "ProofOfWeightsSynapse":
                response = await self.handle_pow_request(synapse_data)
            elif synapse_type == "Competition":
                response = await self.handle_competition_request(synapse_data)
            else:
                response = {
                    "error": f"Unknown synapse type: {synapse_type}",
                    "success": False,
                }

            # Send response
            response_json = json.dumps(response)
            self._quic.send_stream_data(
                0, response_json.encode("utf-8"), end_stream=True
            )

            bt.logging.info(f"✅ Sent response for {synapse_type}")

        except Exception as e:
            bt.logging.error(f"Error processing synapse: {e}")
            error_response = {"error": str(e), "success": False}
            error_json = json.dumps(error_response)
            self._quic.send_stream_data(0, error_json.encode("utf-8"), end_stream=True)

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
                if hasattr(synapse, key):
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
                if hasattr(synapse, key):
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
                if hasattr(synapse, key):
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

            # Create QUIC configuration
            configuration = QuicConfiguration(
                alpn_protocols=["lightning-quic"],
                is_client=False,
            )

            # Create self-signed certificate for QUIC
            configuration.load_cert_chain("cert.pem", "key.pem")

            # Start server
            def create_protocol(*args, **kwargs):
                kwargs["miner_session"] = self.miner_session
                return LightningMinerProtocol(*args, **kwargs)

            self.server = await serve(
                self.bind_address,
                self.port,
                configuration=configuration,
                create_protocol=create_protocol,
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
