from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import json
import toml


class CircuitType(str, Enum):
    """
    Enum representing the type of circuit.
    """

    PROOF_OF_WEIGHTS = "proof_of_weights"
    PROOF_OF_COMPUTATION = "proof_of_computation"


class ProofSystem(str, Enum):
    """
    Enum representing supported proof systems.
    """

    # Supported provers
    ZKML = "ZKML"
    CIRCOM = "CIRCOM"
    EZKL = "EZKL"
    DCAP = "DCAP"

    def __str__(self):
        return self.value

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            return cls(value.upper())
        raise ValueError(f"Cannot convert {value} to {cls.__name__}")

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, value):
        return cls(value)


@dataclass
class CircuitMetadata:
    """
    Metadata for a specific model, such as name, version, description, etc.
    """

    name: str
    description: str
    author: str
    version: str
    proof_system: ProofSystem
    type: CircuitType
    external_files: dict[str, str]
    netuid: int | None = None
    weights_version: int | None = None
    timeout: int | None = None
    benchmark_choice_weight: float | None = None

    @classmethod
    def from_file(cls, metadata_path: str) -> CircuitMetadata:
        """
        Create a ModelMetadata instance from a JSON file.

        Args:
            metadata_path (str): Path to the metadata JSON file.

        Returns:
            ModelMetadata: An instance of ModelMetadata.
        """
        with open(metadata_path, "r", encoding="utf-8") as f:
            if metadata_path.endswith(".json"):
                metadata = json.load(f)
            elif metadata_path.endswith(".toml"):
                metadata = toml.load(f)
            else:
                metadata = {}

        if "proof_system" in metadata:
            metadata["proof_system"] = ProofSystem(metadata["proof_system"])
        if "type" in metadata:
            metadata["type"] = CircuitType(metadata["type"])

        return cls(**metadata)
