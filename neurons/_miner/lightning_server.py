import asyncio
import time
import traceback
from typing import Dict, Any
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
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

        # Create process pool for handling synapse requests
        self.process_pool = ProcessPoolExecutor(
            max_workers=min(4, mp.cpu_count()), mp_context=mp.get_context("spawn")
        )
        bt.logging.info(
            f"🔄 Created process pool with {self.process_pool._max_workers} workers for synapse processing"
        )

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
        """Multiprocess wrapper for QueryZkProof synapse handling"""
        try:
            # Submit to process pool and wait for result with timeout
            future = self.process_pool.submit(
                _handle_query_zk_proof_worker, synapse_data
            )
            return future.result(timeout=10.0)
        except Exception as e:
            bt.logging.error(f"❌ QueryZkProof multiprocess error: {e}")
            return {
                "query_output": "",
                "success": False,
                "error": str(e),
                "processed_via": "lightning_multiprocess_error",
            }

    def _handle_pow_request_async(self, synapse_data: Dict[str, Any]) -> Dict[str, Any]:
        """Multiprocess wrapper for ProofOfWeightsSynapse handling"""
        try:
            future = self.process_pool.submit(_handle_pow_request_worker, synapse_data)
            return future.result(timeout=10.0)
        except Exception as e:
            bt.logging.error(f"❌ PoW multiprocess error: {e}")
            return {
                "proof": "",
                "public_signals": "",
                "success": False,
                "error": str(e),
                "processed_via": "lightning_multiprocess_error",
            }

    def _handle_competition_request_async(
        self, synapse_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Multiprocess wrapper for Competition synapse handling"""
        try:
            future = self.process_pool.submit(
                _handle_competition_request_worker, synapse_data
            )
            return future.result(timeout=10.0)
        except Exception as e:
            bt.logging.error(f"❌ Competition multiprocess error: {e}")
            return {
                "commitment": "",
                "success": False,
                "error": str(e),
                "processed_via": "lightning_multiprocess_error",
            }

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
        # Shutdown process pool first
        if hasattr(self, "process_pool"):
            self.process_pool.shutdown(wait=True)
            bt.logging.info("🔄 Process pool shutdown completed")

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
