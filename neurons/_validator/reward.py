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
    """
    Shifted tan curve
    `x` is a value between 0 and 1
    Output is a value between approximately -6.3 and 6.3
    x = 0.5 gives 0
    x < 0.5 gives negative output
    x > 0.5 gives positive output
    Graph of the function - https://www.desmos.com/calculator/dpfikiniz8
    """
    return math.tan((x - 0.5) * math.pi * FLATTENING_COEFFICIENT)


def g(x):
    """
    `x` is a value between 0 and 1
    The result is a value between 0 and 12 approximately
    Output increases with `x`
    Creates a smooth curve for the reward function
    The graph of the function - https://www.desmos.com/calculator/vklt2b5arm
    """
    return f(x) - f(0)


def h(x):
    """
    `x` is a value between 0 and 1
    The output is also a value between 0 and 1 with positive non-linear mapping
    The graph of the function https://www.desmos.com/calculator/y5ebhrrjjr
    """
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
        # Normalizes response_score based on the response time, ranging from 0 to 0.99
        # larger response time leads to larger response_score
        # The larger the response score, the worse the response time
        response_score = min(
            max(response_time - min_response_time, 0)
            / (max_response_time - min_response_time),
            0.99,
        )
        # Converts the response score to a performance metric.
        # Performance metric has negative correlation with the score (and response time accordingly)
        # but with a non-linear mapping. The metric is a value between 0 and 1.
        # Graph - https://www.desmos.com/calculator/6vogdkyrcj (time weight is one, proof size weight is zero)
        performance_metric = RESPONSE_TIME_WEIGHT * (
            1 - h(response_score)
        ) - PROOF_SIZE_WEIGHT * min(1, proof_size / PROOF_SIZE_THRESHOLD)
        # Fixed rate of change
        rate = RATE_OF_RECOVERY
        # Better performance -> larger max_score
        max_score = max_score * performance_metric
        # Difference in current score vs max_score
        distance = max_score - score
        # Increase score based on rate of change and distance away from a target max_score
        return score + rate * distance

    bt.logging.trace(f"Decaying score {score}")

    return score - rate * distance
