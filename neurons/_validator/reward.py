"""
This module contains the reward function for the validator.
"""

import bittensor as bt

RATE_OF_RECOVERY = 0.2
RATE_OF_DECAY = 0.8

MINIMUM_SCORE = 0

RESPONSE_TIME_WEIGHT = 0.2
PROOF_SIZE_WEIGHT = 0.1

RESPONSE_TIME_THRESHOLD = 40
PROOF_SIZE_THRESHOLD = 30000


import torch
import torch.nn as nn


class Reward(nn.Module):
    def __init__(self):
        super(Reward, self).__init__()
        self.rate_of_decay = RATE_OF_DECAY
        self.rate_of_recovery = RATE_OF_RECOVERY
        self.minimum_score = MINIMUM_SCORE

    def forward(self, max_score, score, verification_result, factor):
        """
        This method calculates the reward for a miner based on the provided score, verification_result, and factor using a neural network module.
        Positional Arguments:
            max_score (Tensor): The maximum score for the miner.
            score (Tensor): The current score for the miner.
            verification_result (Tensor): Whether the response that the miner submitted was valid. (1 or 0)
            factor (Tensor): The factor to apply to the reward, in case the miner is using multiple hotkeys or serving from the same IP multiple times.
        Returns:
            Tensor: The new score for the miner.
        """
        # Use 'verification_result' to switch between recovery and decay rates
        rate = (
            self.rate_of_decay * (1 - verification_result)
            + self.rate_of_recovery * verification_result
        )

        # Calculate distance based on 'verification_result'
        distance = (max_score - score) * verification_result + (
            score - self.minimum_score
        ) * (1 - verification_result)

        # Apply factor
        new_score = score + rate * distance * factor

        return new_score
