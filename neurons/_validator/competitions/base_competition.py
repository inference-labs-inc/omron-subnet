from __future__ import annotations
from abc import ABC, abstractmethod
import torch
import os
from neurons.execution_layer.circuit import Circuit


class BaseCompetition(ABC):
    """
    Base class for competitions.
    """

    def __init__(self, competition_id: int):
        self.competition_id: int = competition_id
        self.competition_directory: str = os.path.join(
            "competitions", str(competition_id)
        )
        self.baseline_model: torch.nn.Module = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        """
        Load the baseline model.
        """
        model = torch.load(os.path.join(self.competition_directory, "model.pth"))
        return model

    @abstractmethod
    def evaluate_circuit(self, circuit: Circuit) -> float:
        """
        Evaluate a circuit and return metrics.
        """
        pass

    def compare_outputs(self, expected: list[float], actual: list[float]) -> float:
        """
        Compare expected and actual outputs and return a score.
        """
        return (expected == actual).mean()
