from __future__ import annotations
import bittensor as bt
from rich.console import Console
from rich.table import Table

from constants import (
    ONE_MINUTE,
    NUM_MINER_GROUPS,
    EPOCH_TEMPO,
    MINER_RESET_WINDOW_BLOCKS,
    FIVE_MINUTES,
)
from utils.rate_limiter import with_rate_limit
from utils.epoch import get_epoch_start_block
from utils import wandb_logger


class ResetManager:
    def __init__(self, metagraph: bt.metagraph):
        self.metagraph = metagraph
        self.reset_tracker = [True for _ in range(len(self.metagraph.uids))]

    @with_rate_limit(period=FIVE_MINUTES)
    def log_reset_tracker(self):
        table = Table(title="Reset Tracker")
        table.add_column("UID", justify="center", style="cyan")
        table.add_column("Reset", justify="center", style="magenta")
        for uid, reset in enumerate(self.reset_tracker):
            table.add_row(str(uid), "✅" if not reset else "❌")
        console = Console()
        console.print(table)
        wandb_logger.safe_log(
            {
                "reset_tracker": {
                    uid: int(value) for uid, value in enumerate(self.reset_tracker)
                }
            }
        )

    @with_rate_limit(period=ONE_MINUTE)
    def _get_last_bonds_submissions(self) -> dict[str, int]:
        """Get the latest bonds submissions and format them into a dict."""
        raw_submissions = self.metagraph.subtensor.substrate.query_map(
            "Commitments",
            "LastBondsReset",
            params=[self.metagraph.netuid],
        )

        submissions = {}
        if not raw_submissions:
            return submissions

        for key_tuple, block_number_scale in raw_submissions:
            try:
                hotkey_bytes = bytes(key_tuple[0])
                hotkey_ss58 = bt.Keypair(public_key=hotkey_bytes.hex()).ss58_address
                submissions[hotkey_ss58] = block_number_scale.value
            except Exception as e:
                bt.logging.warning(
                    f"Could not decode hotkey from LastBondsReset storage map: {e}"
                )

        return submissions

    def miner_missed_reset(
        self,
        uid: int,
        miner_group: int,
        current_epoch: int,
        blocks_until_next_epoch: int,
    ) -> bool:
        """
        Check if a miner missed their required reset submission during their tempo.
        Returns True if the miner was supposed to reset but didn't.
        """
        try:
            last_bonds_submissions = self._get_last_bonds_submissions()

            if self.metagraph.hotkeys[uid] not in last_bonds_submissions:
                return True

            last_reset_block = last_bonds_submissions[self.metagraph.hotkeys[uid]]

            current_active_group = current_epoch % NUM_MINER_GROUPS
            if miner_group == current_active_group:
                most_recent_group_epoch = current_epoch - NUM_MINER_GROUPS
            else:
                epochs_since_last = (
                    current_active_group - miner_group + NUM_MINER_GROUPS
                ) % NUM_MINER_GROUPS
                most_recent_group_epoch = current_epoch - epochs_since_last

            # Do not penalize miners for resets before deployment
            if most_recent_group_epoch < 5744188:
                return False

            if most_recent_group_epoch >= 0:
                epoch_start_block = get_epoch_start_block(
                    most_recent_group_epoch, self.metagraph.netuid
                )
                epoch_end_block = epoch_start_block + EPOCH_TEMPO
                if (
                    last_reset_block < (epoch_end_block - MINER_RESET_WINDOW_BLOCKS)
                    or last_reset_block > epoch_end_block
                ):
                    return True

            return False
        except Exception as e:
            bt.logging.error(f"Error checking reset status for miner {uid}: {e}")
            return False

    def get_reset_tracker(self) -> list[bool]:
        return self.reset_tracker

    def set_reset_status(self, uid: int, status: bool):
        self.reset_tracker[uid] = status
