"""
This module contains the reward function for the validator.
"""

import math

import bittensor as bt

RATE_OF_RECOVERY = 0.1
RATE_OF_DECAY = 0.4

MINIMUM_SCORE = 0

RESPONSE_TIME_WEIGHT = 1
PROOF_SIZE_WEIGHT = 0

PROOF_SIZE_THRESHOLD = 30000

# Controls how flat the curve is
FLATTENING_COEFFICIENT = 9 / 10


def f(x):
    return math.tan((x - 0.5) * math.pi * FLATTENING_COEFFICIENT)


def g(x):
    return f(x) - f(0)


def h(x):
    return g(x) / g(1)


def reward(
    max_score,
    score,
    value,
    response_time,
    proof_size,
    max_response_time,
    min_response_time,
):
    """
    This function calculates the reward for a miner based on the provided score, value, response time and proof size.
    Positional Arguments:
        max_score (int): The maximum score for the miner.
        score (int): The current score for the miner.
        value (bool): Whether the response that the miner submitted was valid.
        response_time (float): The time taken to respond to the query.
        max_response_time (float): The maximum response time across all miners.
        min_response_time (float): The minimum response time across all miners.
    Returns:
        int: The new score for the miner.
    """
    rate = RATE_OF_DECAY
    distance = score - MINIMUM_SCORE
    max_response_time = min(max(max_response_time, 0), 30)
    min_response_time = min(max(min_response_time, 0), max_response_time)
    if value:
        response_score = min(
            max(response_time - min_response_time, 0)
            / (max_response_time - min_response_time),
            0.99,
        )
        performance_metric = RESPONSE_TIME_WEIGHT * (
            1 - h(response_score)
        ) - PROOF_SIZE_WEIGHT * min(1, proof_size / PROOF_SIZE_THRESHOLD)
        rate = RATE_OF_RECOVERY * performance_metric
        max_score = max_score * performance_metric
        distance = max_score - score
        return score + rate * distance

    bt.logging.trace(f"Decaying score {score}")

    return score - rate * distance
