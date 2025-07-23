import asyncio
import time
import traceback
from typing import Dict, Any
import bittensor as bt
from lightning.lightning import RustLightningServer


class LightningServer:
    """Pure Rust Lightning server - no Python fallback"""

    def __init__(self, miner_session, host: str = "0.0.0.0", port: int = 8092):
        self.miner_session = miner_session
        self.host = host
        self.port = port
        self.connection_cleanup_interval = 300
        self.stats_log_interval = 120
        self.last_stats_log = 0

        miner_hotkey = getattr(
            miner_session.wallet.hotkey, "ss58_address", "unknown_miner"
        )
        self.rust_server = RustLightningServer(miner_hotkey, host, port)
        self._register_synapse_handlers()
        bt.logging.success("⚡ Rust Lightning server initialized")

    def _register_synapse_handlers(self):
        """Register synapse handlers with the Lightning server"""
        self.rust_server.register_synapse_handler(
            "QueryZkProof", self._handle_query_zk_proof
        )
        self.rust_server.register_synapse_handler(
            "ProofOfWeightsSynapse", self._handle_pow_request
        )
        self.rust_server.register_synapse_handler(
            "Competition", self._handle_competition_request
        )
        bt.logging.success("📝 Registered all synapse handlers with Lightning server")

    def _handle_query_zk_proof(self, synapse_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle QueryZkProof synapse through Lightning server"""
        try:
            from protocol import QueryZkProof

            synapse = QueryZkProof()
            for key, value in synapse_data.items():
                if hasattr(synapse, key):
                    setattr(synapse, key, value)

            result_synapse = self.miner_session.queryZkProof(synapse)

            return {
                "query_output": getattr(result_synapse, "query_output", ""),
                "success": True,
                "processed_via": "lightning_server",
            }
        except Exception as e:
            bt.logging.error(f"❌ Lightning QueryZkProof handler error: {e}")
            return {
                "query_output": "",
                "success": False,
                "error": str(e),
                "processed_via": "lightning_server_error",
            }

    def _handle_pow_request(self, synapse_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ProofOfWeightsSynapse through Lightning server"""
        try:
            from protocol import ProofOfWeightsSynapse

            synapse = ProofOfWeightsSynapse()
            for key, value in synapse_data.items():
                if hasattr(synapse, key):
                    setattr(synapse, key, value)

            result_synapse = self.miner_session.handle_pow_request(synapse)

            return {
                "proof": getattr(result_synapse, "proof", ""),
                "public_signals": getattr(result_synapse, "public_signals", ""),
                "success": True,
                "processed_via": "lightning_server",
            }
        except Exception as e:
            bt.logging.error(f"❌ Lightning PoW handler error: {e}")
            return {
                "proof": "",
                "public_signals": "",
                "success": False,
                "error": str(e),
                "processed_via": "lightning_server_error",
            }

    def _handle_competition_request(
        self, synapse_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle Competition synapse through Lightning server"""
        try:
            from protocol import Competition

            synapse = Competition()
            for key, value in synapse_data.items():
                if hasattr(synapse, key):
                    setattr(synapse, key, value)

            result_synapse = self.miner_session.handleCompetitionRequest(synapse)

            return {
                "commitment": getattr(result_synapse, "commitment", ""),
                "success": True,
                "processed_via": "lightning_server",
            }
        except Exception as e:
            bt.logging.error(f"❌ Lightning Competition handler error: {e}")
            return {
                "commitment": "",
                "success": False,
                "error": str(e),
                "processed_via": "lightning_server_error",
            }

    async def start(self) -> None:
        """Start the Lightning server"""
        self.rust_server.start()
        bt.logging.success(f"⚡ Lightning server started on {self.host}:{self.port}")
        bt.logging.info("🔐 Lightning server implements optimized handshake protocol")
        bt.logging.info(
            "📦 Lightning server ready for persistent validator connections"
        )

        asyncio.create_task(self._periodic_cleanup())

    async def _periodic_cleanup(self):
        """Periodically cleanup stale connections and log stats"""
        while True:
            try:
                await asyncio.sleep(self.connection_cleanup_interval)

                self.rust_server.cleanup_stale_connections(600)

                current_time = time.time()
                if current_time - self.last_stats_log > self.stats_log_interval:
                    stats = self.rust_server.get_connection_stats()
                    active_connections = stats.get("verified_connections", 0)
                    bt.logging.info(
                        f"⚡ Lightning server: {active_connections} active validator connections"
                    )
                    self.last_stats_log = current_time

            except Exception as e:
                bt.logging.error(f"❌ Lightning server cleanup error: {e}")

    async def stop(self) -> None:
        """Stop the Lightning server"""
        self.rust_server.stop()
        bt.logging.info("🔌 Lightning server stopped, all connections closed")

    async def serve_forever(self) -> None:
        """Serve forever (for compatibility with existing code)"""
        bt.logging.info("⚡ Lightning server running in persistent connection mode")
        bt.logging.info("🚀 Ready to handle high-throughput validator requests")

        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            bt.logging.debug("Lightning server serve_forever cancelled")
        except Exception as e:
            bt.logging.error(f"❌ Lightning server error: {e}")
            traceback.print_exc()

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get Lightning server connection statistics"""
        stats = self.rust_server.get_connection_stats()
        stats["lightning_available"] = True
        stats["server_type"] = "lightning_rust"
        return stats


def create_lightning_server(miner_session, host: str = "0.0.0.0", port: int = 8092):
    """Factory function to create Lightning server"""
    return LightningServer(miner_session, host, port)
