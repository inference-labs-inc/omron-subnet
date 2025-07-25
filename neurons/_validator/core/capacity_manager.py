import bittensor as bt
from _validator.config import ValidatorConfig
from protocol import QueryForCapacities


class CapacityManager:
    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.dendrite = bt.dendrite(wallet=self.config.wallet)

    async def sync_capacities(self, axons: list[bt.Axon]):
        bt.logging.info(f"Syncing capacities for {len(axons)} axons")
        query_coroutine = self.dendrite.query(axons, QueryForCapacities())
        if query_coroutine is None:
            bt.logging.warning(
                "Dendrite query returned None, possibly not running. Returning empty list."
            )
            return []
        return await query_coroutine
