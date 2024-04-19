"""
This module contains the reward function for the validator.
"""

import bittensor as bt

RATE_OF_RECOVERY = 0.2
RATE_OF_DECAY = 0.4

MINIMUM_SCORE = 0

RESPONSE_TIME_WEIGHT = 0.2
PROOF_SIZE_WEIGHT = 0.1

RESPONSE_TIME_THRESHOLD = 240
PROOF_SIZE_THRESHOLD = 30000


import torch
import torch.nn as nn


class Reward(nn.Module):
    def __init__(self):
        super(Reward, self).__init__()
        self.rate_of_decay = RATE_OF_DECAY
        self.rate_of_recovery = RATE_OF_RECOVERY
        self.minimum_score = MINIMUM_SCORE
        self.response_time_weight = RESPONSE_TIME_WEIGHT
        self.proof_size_weight = PROOF_SIZE_WEIGHT
        self.response_time_threshold = RESPONSE_TIME_THRESHOLD
        self.proof_size_threshold = PROOF_SIZE_THRESHOLD

    def forward(self, max_score, score, verification_result, response_time, proof_size):
        """
        This method calculates the reward for a miner based on the provided score, verification_result, and factor using a neural network module.
        Positional Arguments:
            max_score (Tensor): The maximum score for the miner.
            score (Tensor): The current score for the miner.
            verification_result (Tensor): Whether the response that the miner submitted was valid. (1 or 0)
            response_time (Tensor): The time taken to respond to the query.
            proof_size (Tensor): The size of the proof.
        Returns:
            Tensor: The new score for the miner.
        """
        performance_metric = (
            1
            - self.response_time_weight
            * torch.min(torch.tensor(1.0), response_time / self.response_time_threshold)
            - self.proof_size_weight
            * torch.min(torch.tensor(1.0), proof_size / self.proof_size_threshold)
        )

        rate = (
            self.rate_of_recovery * performance_metric * verification_result
            + self.rate_of_decay * (1 - verification_result)
        )
        distance = (max_score - score) * verification_result + (
            score - self.minimum_score
        ) * (1 - verification_result)
        new_score = (
            score
            + rate * distance
            - (1 - performance_metric) * 0.005 * verification_result
        )
        return torch.min(torch.tensor(1.0), torch.max(torch.tensor(0.0), new_score))
