"""
This module contains the reward function for the validator.
"""

import bittensor as bt

RATE_OF_RECOVERY = 0.2
RATE_OF_DECAY = 0.8
MINIMUM_SCORE = 0


def reward(max_score, score, value, factor):
    """
    This function calculates the reward for a miner based on the provided score, value, and factor.
    Positional Arguments:
        max_score (int): The maximum score for the miner.
        score (int): The current score for the miner.
        value (bool): Whether the response that the miner submitted was valid.
        factor (float): The factor to apply to the reward, in case the miner is using multiple hotkeys or serving from the same IP multiple times.
    Returns:
        int: The new score for the miner.
    """
    rate = RATE_OF_DECAY
    distance = score - MINIMUM_SCORE
    if value:
        bt.logging.trace(f"Recovering score {score}")
        rate = RATE_OF_RECOVERY
        distance = max_score - score
        return score + rate * distance * factor
    else:
        bt.logging.trace(f"Decaying score {score}")
        return score - rate * distance * factor
