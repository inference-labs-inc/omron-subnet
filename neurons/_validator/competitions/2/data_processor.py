import torch
from ..services.data_source import CompetitionDataProcessor


class ImageClassificationProcessor(CompetitionDataProcessor):
    def __init__(self, noise_scale: float = 0.02, jitter_scale: float = 0.1):
        self.noise_scale = noise_scale
        self.jitter_scale = jitter_scale

    def process_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        # Add random noise
        noise = torch.randn_like(inputs) * self.noise_scale
        perturbed = inputs + noise

        # Random color jitter
        jitter = torch.randn(3, 1, 1) * self.jitter_scale
        perturbed = perturbed + jitter.expand_as(perturbed)

        # Random horizontal flip
        if torch.rand(1) > 0.5:
            perturbed = torch.flip(perturbed, dims=[3])

        # Ensure values stay in valid range [0, 1]
        perturbed = torch.clamp(perturbed, 0, 1)

        return perturbed
