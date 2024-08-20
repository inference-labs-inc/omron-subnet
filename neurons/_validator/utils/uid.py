from collections.abc import Generator, Iterable
import bittensor as bt
import torch

from constants import VALIDATOR_STAKE_THRESHOLD


def get_queryable_uids(
    metagraph: bt.metagraph, uids: list[int]
) -> Generator[int, None, None]:
    """
    Returns the uids of the miners that are queryable
    """
    # Ignore validators, they're not queryable as miners (torch.nn.Parameter)
    queryable_flags: Iterable[bool] = (
        (metagraph.total_stake[uids] < VALIDATOR_STAKE_THRESHOLD)
        & torch.tensor([metagraph.axons[i].ip != "0.0.0.0" for i in uids])
    ).tolist()
    for uid, is_queryable in zip(uids, queryable_flags):
        if is_queryable:
            yield uid
