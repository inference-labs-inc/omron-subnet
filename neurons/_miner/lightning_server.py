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

        # Temporarily disable multiprocessing to debug startup issues
        bt.logging.info(
            "🔄 Using synchronous processing (multiprocessing disabled for debugging)"
        )
        self.process_pool = None

        miner_hotkey = getattr(
            miner_session.wallet.hotkey, "ss58_address", "unknown_miner"
        )
        self.rust_server = RustLightningServer(miner_hotkey, host, port)
        self._register_synapse_handlers()
        bt.logging.success("⚡ Rust Lightning server initialized")

    def _register_synapse_handlers(self):
        """Register synapse handlers with the Lightning server"""
        self.rust_server.register_synapse_handler(
            "QueryZkProof", self._handle_query_zk_proof_async
        )
        self.rust_server.register_synapse_handler(
            "ProofOfWeightsSynapse", self._handle_pow_request_async
        )
        self.rust_server.register_synapse_handler(
            "Competition", self._handle_competition_request_async
        )
        bt.logging.success(
            "📝 Registered all synapse handlers with Lightning server (using multiprocessing)"
        )

    def _handle_query_zk_proof_async(
        self, synapse_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle QueryZkProof directly - no multiprocessing"""
        print("🔥 HANDLER CALLED - QueryZkProof Python handler executing!")
        bt.logging.critical("🔥 CRITICAL: QueryZkProof Python handler is executing!")

        return self._handle_query_zk_proof(synapse_data)

    def _handle_pow_request_async(self, synapse_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ProofOfWeightsSynapse directly - no multiprocessing"""
        print("🔥 HANDLER CALLED - PoW Python handler executing!")
        bt.logging.critical("🔥 CRITICAL: PoW Python handler is executing!")

        return self._handle_pow_request(synapse_data)

    def _handle_competition_request_async(
        self, synapse_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle Competition directly - no multiprocessing"""
        print("🔥 HANDLER CALLED - Competition Python handler executing!")
        bt.logging.critical("🔥 CRITICAL: Competition Python handler is executing!")

        return self._handle_competition_request(synapse_data)

    def _handle_query_zk_proof(self, synapse_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle QueryZkProof synapse through Lightning server"""
        print("🔥 CORE HANDLER CALLED - Core Python handler executing!")
        bt.logging.critical("🔥 CRITICAL: Core Python handler is executing!")

        try:
            bt.logging.info("🔧 Query Handler: Starting core processing")
            bt.logging.info(
                f"📊 Query Handler: Received {len(synapse_data)} data fields"
            )

            # Just return a simple test response without calling miner_session
            result = {
                "query_output": "TEST_QUERY_RESPONSE_FROM_LIGHTNING",
                "success": True,
                "processed_via": "lightning_server_test",
            }

            bt.logging.success("🎉 Query Handler: SUCCESS - returning test response")
            return result
        except Exception as e:
            bt.logging.error(f"❌ Lightning QueryZkProof handler error: {e}")
            import traceback

            bt.logging.debug(f"❌ Query Handler traceback: {traceback.format_exc()}")
            return {
                "query_output": "",
                "success": False,
                "error": str(e),
                "processed_via": "lightning_server_error",
            }

    def _handle_pow_request(self, synapse_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ProofOfWeightsSynapse through Lightning server"""
        print("🔥 CORE HANDLER CALLED - PoW Core Python handler executing!")
        bt.logging.critical("🔥 CRITICAL: PoW Core Python handler is executing!")

        try:
            bt.logging.info("🔧 PoW Handler: Starting core processing")
            bt.logging.info(f"📊 PoW Handler: Received {len(synapse_data)} data fields")

            # Just return a simple test response without calling miner_session
            result = {
                "proof": "TEST_PROOF_FROM_LIGHTNING",
                "public_signals": "TEST_SIGNALS_FROM_LIGHTNING",
                "success": True,
                "processed_via": "lightning_server_test",
            }

            bt.logging.success("🎉 PoW Handler: SUCCESS - returning test response")
            return result
        except Exception as e:
            bt.logging.error(f"❌ Lightning PoW handler error: {e}")
            import traceback

            bt.logging.debug(f"❌ PoW Handler traceback: {traceback.format_exc()}")
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
        print("🔥 CORE HANDLER CALLED - Competition Core Python handler executing!")
        bt.logging.critical(
            "🔥 CRITICAL: Competition Core Python handler is executing!"
        )

        try:
            bt.logging.info("🔧 Competition Handler: Starting core processing")
            bt.logging.info(
                f"📊 Competition Handler: Received {len(synapse_data)} data fields"
            )

            # Just return a simple test response without calling miner_session
            result = {
                "commitment": "TEST_COMMITMENT_FROM_LIGHTNING",
                "success": True,
                "processed_via": "lightning_server_test",
            }

            bt.logging.success(
                "🎉 Competition Handler: SUCCESS - returning test response"
            )
            return result
        except Exception as e:
            bt.logging.error(f"❌ Lightning Competition handler error: {e}")
            import traceback

            bt.logging.debug(
                f"❌ Competition Handler traceback: {traceback.format_exc()}"
            )
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
        # Shutdown process pool first
        if hasattr(self, "process_pool") and self.process_pool is not None:
            self.process_pool.shutdown(wait=True)
            bt.logging.info("🔄 Process pool shutdown completed")

        self.rust_server.stop()
        bt.logging.info("🔌 Lightning server stopped, all connections closed")

    async def serve_forever(self) -> None:
        """Start the actual Rust QUIC server listening loop"""
        bt.logging.info("⚡ Lightning server running in persistent connection mode")
        bt.logging.info("🚀 Ready to handle high-throughput validator requests")

        try:
            # Call the Rust server's serve_forever method (blocking call)
            await asyncio.get_event_loop().run_in_executor(
                None, self.rust_server.serve_forever
            )
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


# Worker functions for multiprocessing (must be at module level for pickling)
def _handle_query_zk_proof_worker(synapse_data: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function for QueryZkProof synapse handling in separate process"""
    try:
        from protocol import QueryZkProof

        # Import here to avoid pickling issues
        import os
        import sys

        # Add the current directory to path to ensure imports work
        sys.path.insert(0, os.getcwd())

        # Create a minimal synapse object just for processing
        synapse = QueryZkProof()
        for key, value in synapse_data.items():
            if hasattr(synapse, key):
                setattr(synapse, key, value)

        # This is a simplified version - in real implementation, we'd need
        # to recreate the miner session or find another way to process
        # For now, return a dummy response to test the multiprocessing
        return {
            "query_output": "dummy_output_from_multiprocess",
            "success": True,
            "processed_via": "lightning_multiprocess",
        }
    except Exception as e:
        return {
            "query_output": "",
            "success": False,
            "error": str(e),
            "processed_via": "lightning_multiprocess_error",
        }


def _handle_pow_request_worker(synapse_data: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function for ProofOfWeightsSynapse handling in separate process"""
    try:
        from protocol import ProofOfWeightsSynapse

        synapse = ProofOfWeightsSynapse()
        for key, value in synapse_data.items():
            if hasattr(synapse, key):
                setattr(synapse, key, value)

        return {
            "proof": "dummy_proof_from_multiprocess",
            "public_signals": "dummy_signals_from_multiprocess",
            "success": True,
            "processed_via": "lightning_multiprocess",
        }
    except Exception as e:
        return {
            "proof": "",
            "public_signals": "",
            "success": False,
            "error": str(e),
            "processed_via": "lightning_multiprocess_error",
        }


def _handle_competition_request_worker(synapse_data: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function for Competition synapse handling in separate process"""
    try:
        from protocol import Competition

        synapse = Competition()
        for key, value in synapse_data.items():
            if hasattr(synapse, key):
                setattr(synapse, key, value)

        return {
            "commitment": "dummy_commitment_from_multiprocess",
            "success": True,
            "processed_via": "lightning_multiprocess",
        }
    except Exception as e:
        return {
            "commitment": "",
            "success": False,
            "error": str(e),
            "processed_via": "lightning_multiprocess_error",
        }
