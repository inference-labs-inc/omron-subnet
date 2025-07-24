import time
from typing import Dict, Any
import bittensor as bt
from lightning.lightning import RustLightning


class LightningTransport:
    """Pure Rust Lightning transport - no Python fallback"""

    def __init__(self, wallet):
        self.wallet = wallet
        self.connection_stats_log_interval = 60
        self.last_stats_log = 0

        wallet_hotkey = getattr(wallet.hotkey, "ss58_address", "unknown_hotkey")
        self.lightning_client = RustLightning(wallet_hotkey)

        # Configure validator keypair for real signature generation
        try:
            # Extract seed from bittensor wallet for signing
            if hasattr(wallet.hotkey, "seed_hex"):
                # Get seed from hex string
                seed_hex = wallet.hotkey.seed_hex
                if seed_hex and len(seed_hex) == 64:  # 32 bytes as hex
                    seed_bytes = bytes.fromhex(seed_hex)
                    self.lightning_client.set_validator_keypair(list(seed_bytes))
                    bt.logging.success(
                        "🔑 Validator keypair configured for Lightning signatures"
                    )
                else:
                    bt.logging.warning("⚠️ Invalid seed format - using dummy signatures")
            elif hasattr(wallet.hotkey, "private_key_bytes"):
                # Try accessing private key bytes directly
                private_key_bytes = wallet.hotkey.private_key_bytes
                if (
                    isinstance(private_key_bytes, bytes)
                    and len(private_key_bytes) == 32
                ):
                    self.lightning_client.set_validator_keypair(list(private_key_bytes))
                    bt.logging.success(
                        "🔑 Validator keypair configured for Lightning signatures"
                    )
                else:
                    bt.logging.warning(
                        "⚠️ Invalid private key format - using dummy signatures"
                    )
            else:
                bt.logging.warning(
                    "⚠️ No seed/private key available - using dummy signatures"
                )
                bt.logging.debug(
                    f"Available wallet.hotkey attributes: {dir(wallet.hotkey)}"
                )
        except Exception as e:
            bt.logging.warning(f"⚠️ Failed to configure validator keypair: {e}")
            import traceback

            bt.logging.debug(f"Keypair extraction error: {traceback.format_exc()}")

        bt.logging.success("⚡ Rust Lightning client initialized")

    async def initialize_persistent_connections(self, metagraph):
        """Initialize persistent connections to all miners on startup"""
        miners = []
        for uid, axon in enumerate(metagraph.axons):
            if axon and hasattr(axon, "ip") and hasattr(axon, "port"):
                miner_info = {
                    "hotkey": (
                        axon.hotkey if hasattr(axon, "hotkey") else f"miner_{uid}"
                    ),
                    "ip": axon.ip,
                    "port": int(axon.port) + 1,
                    "protocol": 4,
                    "placeholder1": 0,
                    "placeholder2": 0,
                }
                miners.append(miner_info)

        if miners:
            self.lightning_client.initialize_connections(miners)
            bt.logging.success(
                f"⚡ Initialized persistent connections to {len(miners)} miners"
            )

            stats = self.lightning_client.get_connection_stats()
            bt.logging.info(f"📊 Lightning connection stats: {stats}")

    async def update_miner_registry(self, metagraph, force=False):
        """Update miner registry and manage persistent connections"""
        current_time = time.time()
        if (
            not force
            and current_time - getattr(self, "last_miner_registry_update", 0) < 30
        ):
            return

        miners = []
        for uid, axon in enumerate(metagraph.axons):
            if axon and hasattr(axon, "ip") and hasattr(axon, "port"):
                miner_info = {
                    "hotkey": (
                        axon.hotkey if hasattr(axon, "hotkey") else f"miner_{uid}"
                    ),
                    "ip": axon.ip,
                    "port": int(axon.port) + 1,
                    "protocol": 4,
                    "placeholder1": 0,
                    "placeholder2": 0,
                }
                miners.append(miner_info)

        self.lightning_client.update_miner_registry(miners)
        self.last_miner_registry_update = current_time

        if current_time - self.last_stats_log > self.connection_stats_log_interval:
            stats = self.lightning_client.get_connection_stats()
            bt.logging.debug(
                f"⚡ Lightning connections: {stats.get('total_connections', 0)} active"
            )
            self.last_stats_log = current_time

    async def query_axon(
        self, axon, synapse_dict: Dict[str, Any], timeout: float = 12.0
    ) -> Dict[str, Any]:
        """Query axon using Rust Lightning with persistent connections"""
        axon_info = {
            "hotkey": axon.hotkey if hasattr(axon, "hotkey") else "unknown",
            "ip": axon.ip,
            "port": int(axon.port) + 1,
            "protocol": 4,
            "placeholder1": 0,
            "placeholder2": 0,
        }

        synapse_type = synapse_dict.get("synapse_type", "Unknown")

        request = {"synapse_type": synapse_type, "data": synapse_dict.get("data", {})}

        start_time = time.time()
        response = self.lightning_client.query_axon(axon_info, request)
        latency = time.time() - start_time

        if response.get("success", False):
            bt.logging.trace(
                f"⚡ Lightning query to {axon.hotkey[:8]}... completed in {latency:.3f}s"
            )

            result = {
                "success": True,
                "latency_ms": response.get("latency_ms", latency * 1000),
                "status_code": 200,
                "connection_type": "lightning_persistent",
            }

            for key, value in response.items():
                if key not in ["success", "latency_ms"]:
                    result[key] = value

            return result
        else:
            bt.logging.error(f"❌ Lightning query failed for {axon.hotkey[:8]}...")
            raise Exception(
                f"Lightning query failed: {response.get('error', 'Unknown error')}"
            )

    async def close_connections(self):
        """Close all persistent connections"""
        self.lightning_client.close_all_connections()
        bt.logging.info("🔌 Closed all Lightning persistent connections")


def create_lightning_transport(wallet):
    """Factory function to create Lightning transport"""
    return LightningTransport(wallet)
