import bittensor as bt
from _validator.config import ValidatorConfig
from protocol import QueryForCapacities


class CapacityManager:
    def __init__(self, config: ValidatorConfig):
        self.config = config

    def sync_capacities(self, axons: list[bt.Axon]):
        bt.logging.info(f"Syncing capacities for {len(axons)} axons")
        with bt.dendrite(self.config.wallet) as dendrite:
            return dendrite.query(axons, QueryForCapacities())
