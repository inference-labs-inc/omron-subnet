from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import os
import json

# trunk-ignore(pylint/E0611)
from bittensor import logging


class ProofSystem(str, Enum):
    """
    Enum representing supported proof systems.
    """

    # ZK Proof Systems
    ZKML = "ZKML"
    CIRCOM = "CIRCOM"
    JOLT = "JOLT"

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
class CircuitPaths:
    """
    Paths to all files for the provided model.
    """

    model_id: str
    base_path: str = field(init=False)
    input: str = field(init=False)
    metadata: str = field(init=False)
    compiled_model: str = field(init=False)
    pk: str = field(init=False)
    vk: str = field(init=False)
    settings: str = field(init=False)
    witness: str = field(init=False)
    proof: str = field(init=False)
    srs: str = field(init=False)
    witness_executable: str = field(init=False)

    def __post_init__(self):
        self.base_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "deployment_layer",
            f"model_{self.model_id}",
        )
        self.input = os.path.join(self.base_path, "input.json")
        self.metadata = os.path.join(self.base_path, "metadata.json")
        self.compiled_model = os.path.join(self.base_path, "model.compiled")
        self.settings = os.path.join(self.base_path, "settings.json")
        self.witness = os.path.join(self.base_path, "witness.json")
        self.proof = os.path.join(self.base_path, "proof.json")
        self.srs = os.path.join(self.base_path, "kzg.srs")
        self.witness_executable = os.path.join(self.base_path, "witness.js")
        self.pk = os.path.join(self.base_path, "circuit.zkey")
        self.vk = os.path.join(self.base_path, "verification_key.json")

    def set_proof_system_paths(self, proof_system: ProofSystem):
        """
        Set proof system-specific paths.
        """
        if proof_system == ProofSystem.CIRCOM:
            self.pk = os.path.join(self.base_path, "circuit.zkey")
            self.vk = os.path.join(self.base_path, "verification_key.json")
            self.compiled_model = os.path.join(self.base_path, "circuit.wasm")
        elif proof_system == ProofSystem.JOLT:
            self.compiled_model = os.path.join(self.base_path, "circuit")
        else:
            raise ValueError(f"Proof system {proof_system} not supported")


@dataclass
class CircuitMetadata:
    """
    Metadata for a specific model, such as name, version, description, etc.
    """

    name: str
    description: str
    author: str
    version: str
    proof_system: str
    external_files: dict[str, str]

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
            metadata = json.load(f)
        return cls(**metadata)


class Circuit:
    """
    A class representing a circuit.
    """

    def __init__(self, model_id: str):
        """
        Initialize a Model instance.

        Args:
            model_id (str): Unique identifier for the model.
        """
        self.paths = CircuitPaths(model_id)
        self.metadata = CircuitMetadata.from_file(self.paths.metadata)
        self.id = model_id
        self.proof_system = ProofSystem[self.metadata.proof_system]
        self.paths.set_proof_system_paths(self.proof_system)
        self.settings = {}
        try:
            with open(self.paths.settings, "r", encoding="utf-8") as f:
                self.settings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logging.warning(
                f"Failed to load settings for model {self.id}. Using default settings."
            )
