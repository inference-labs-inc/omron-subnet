import bittensor as bt
from _validator.config import ValidatorConfig
import asyncio
from protocol import QueryForCapacities


class CapacityManager:
    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.dendrite = bt.dendrite(self.config.wallet)

    def sync_capacities(self, axons: list[bt.Axon]):
        bt.logging.info(f"Syncing capacities for {len(axons)} axons")
        return asyncio.run(self.dendrite.query(axons, QueryForCapacities()))
