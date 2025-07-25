import bittensor as bt
from _validator.config import ValidatorConfig
from protocol import QueryForCapacities
from utils import with_rate_limit
from constants import ONE_HOUR


class CapacityManager:
    def __init__(self, config: ValidatorConfig):
        self.config = config

    @with_rate_limit(period=ONE_HOUR)
    def sync_capacities(self, axons: list[bt.Axon]):
        with bt.dendrite(self.config.wallet) as dendrite:
            return dendrite.query(axons, QueryForCapacities())
