from __future__ import annotations
from dataclasses import dataclass, field
import torch
import bittensor as bt
from constants import WEIGHT_RATE_LIMIT, WEIGHTS_VERSION
from _validator.utils.logging import log_weights
from _validator.utils.proof_of_weights import ProofOfWeightsItem
from utils.system import timeout_with_multiprocess_retry


@dataclass
class WeightsManager:
    """
    Manages weight setting for the Omron validator.

    Attributes:
        subtensor (bt.subtensor): The Bittensor subtensor instance.
        metagraph (bt.metagraph): The Bittensor metagraph instance.
        wallet (bt.wallet): The Bittensor wallet instance.
        user_uid (int): The unique identifier of the validator.
        weights (Optional[torch.Tensor]): The current weights tensor.
        last_update_weights_block (int): The last block number when weights were updated.
        proof_of_weights_queue (List[ProofOfWeightsItem]): Queue for proof of weights items.
    """

    subtensor: bt.subtensor
    metagraph: bt.metagraph
    wallet: bt.wallet
    user_uid: int
    last_update_weights_block: int = 0
    proof_of_weights_queue: list[ProofOfWeightsItem] = field(default_factory=list)

    def should_update_weights(self) -> bool:
        current_block = self.subtensor.get_current_block()
        bt.logging.trace(f"Current block: {current_block}")
        bt.logging.trace(f"Last update weights block: {self.last_update_weights_block}")
        return current_block - self.last_update_weights_block >= WEIGHT_RATE_LIMIT

    def set_weights(self, netuid, wallet, uids, weights, version_key):
        return self.subtensor.set_weights(
            netuid=netuid,
            wallet=wallet,
            uids=uids,
            weights=weights,
            wait_for_inclusion=True,
            version_key=version_key,
        )

    @timeout_with_multiprocess_retry(seconds=60, retries=3)
    def update_weights(self, scores: torch.Tensor) -> bool:
        """
        Updates the weights based on the given scores and sets them on the chain.

        Args:
            scores (torch.Tensor): The scores tensor used to calculate new weights.
        """
        if not self.should_update_weights():
            current_block = self.subtensor.get_current_block()
            blocks_until_update = WEIGHT_RATE_LIMIT - (
                current_block - self.last_update_weights_block
            )
            minutes_until_update = round((blocks_until_update * 12) / 60, 1)
            bt.logging.info(
                f"Next weight update in {blocks_until_update} blocks (approximately {minutes_until_update:.1f} minutes)"
            )
            return False

        bt.logging.info("Updating weights")
        blocks_since_last_update = self.subtensor.blocks_since_last_update(
            self.metagraph.netuid, self.user_uid
        )
        bt.logging.info(
            f"Blocks since last weights were set: {blocks_since_last_update}"
        )

        weights = torch.zeros(self.metagraph.n)
        nonzero_indices = scores.nonzero()
        bt.logging.debug(
            f"Weights: {weights}, Nonzero indices: {nonzero_indices}, Scores: {scores}"
        )
        if nonzero_indices.sum() > 0:
            weights[nonzero_indices] = scores[nonzero_indices]

        try:
            success, message = self.set_weights(
                netuid=self.metagraph.netuid,
                wallet=self.wallet,
                uids=self.metagraph.uids.tolist(),
                weights=weights.tolist(),
                version_key=WEIGHTS_VERSION,
            )
            if success:
                log_weights(weights)
                self.last_update_weights_block = int(self.metagraph.block.item())
                return True
            new_blocks_since_last_update = self.subtensor.blocks_since_last_update(
                self.metagraph.netuid, self.user_uid
            )
            if new_blocks_since_last_update > blocks_since_last_update:
                bt.logging.success(
                    f"Blocks since last update is now {new_blocks_since_last_update}, "
                    "which is greater than {blocks_since_last_update}. Weights were set."
                )
                return True
            bt.logging.warning("Failed to set weights")
            return False
        except Exception as e:
            bt.logging.warning(f"Failed to set weights on chain with exception: {e}")
            return False
