import torch
from ..services.data_source import CompetitionDataProcessor


class MatrixMultDataProcessor(CompetitionDataProcessor):
    def __init__(self, noise_scale: float = 0.01):
        self.noise_scale = noise_scale

    def process_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(inputs) * self.noise_scale
        perturbed = inputs + noise
        batch_size = inputs.shape[0]
        matrix1 = perturbed[:, :10].reshape(batch_size, 2, 5)
        matrix2 = perturbed[:, 10:].reshape(batch_size, 5, 3)

        matrix1 = torch.clamp(matrix1, -10, 10)
        matrix2 = torch.clamp(matrix2, -10, 10)

        return torch.cat(
            [matrix1.reshape(batch_size, -1), matrix2.reshape(batch_size, -1)], dim=1
        )
