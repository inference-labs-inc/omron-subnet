from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import torch
import os
import json
import cli_parser
from deployment_layer.circuit_store import CircuitStore
from execution_layer.input_registry import InputRegistry

# trunk-ignore(pylint/E0611)
from bittensor import logging
from constants import (
    MAX_EVALUATION_ITEMS,
    DEFAULT_PROOF_SIZE,
    MAXIMUM_SCORE_MEDIAN_SAMPLE,
    CRICUIT_TIMEOUT_SECONDS,
)


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

    # ZK Proof Systems
    ZKML = "ZKML"
    CIRCOM = "CIRCOM"
    JOLT = "JOLT"
    EZKL = "EZKL"

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
        if hasattr(cli_parser, "config") and cli_parser.config.full_path_models:
            self.external_base_path = os.path.join(
                cli_parser.config.full_path_models,
                f"model_{self.model_id}",
            )
        else:
            self.external_base_path = os.path.join(
                os.path.expanduser("~"),
                ".bittensor",
                "omron",
                "models",
                f"model_{self.model_id}",
            )
        self.input = os.path.join(self.base_path, "input.json")
        self.metadata = os.path.join(self.base_path, "metadata.json")
        self.compiled_model = os.path.join(self.base_path, "model.compiled")
        self.settings = os.path.join(self.base_path, "settings.json")
        self.witness = os.path.join(self.base_path, "witness.json")
        self.proof = os.path.join(self.base_path, "proof.json")
        self.witness_executable = os.path.join(self.base_path, "witness.js")
        self.pk = os.path.join(self.external_base_path, "circuit.zkey")
        self.vk = os.path.join(self.base_path, "verification_key.json")
        self.evaluation_data = os.path.join(
            self.external_base_path, "evaluation_data.json"
        )

    def set_proof_system_paths(self, proof_system: ProofSystem):
        """
        Set proof system-specific paths.
        """
        if proof_system == ProofSystem.CIRCOM:
            self.pk = os.path.join(self.external_base_path, "circuit.zkey")
            self.vk = os.path.join(self.base_path, "verification_key.json")
            self.compiled_model = os.path.join(self.base_path, "circuit.wasm")
        elif proof_system == ProofSystem.JOLT:
            self.compiled_model = os.path.join(
                self.base_path, "target", "release", "circuit"
            )
        elif proof_system == ProofSystem.EZKL:
            self.pk = os.path.join(self.external_base_path, "pk.key")
            self.vk = os.path.join(self.base_path, "vk.key")
            self.compiled_model = os.path.join(self.base_path, "model.compiled")
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
            metadata = json.load(f)
        return cls(**metadata)


@dataclass
class CircuitEvaluationItem:
    """
    Data collected from the evaluation of the circuit.
    """

    circuit_id: str = field(default="")
    uid: int = field(default=0)
    minimum_response_time: float = field(default=0.0)
    maximum_response_time: float = field(init=False) # Field will be set in __post_init__
    proof_size: int = field(default=DEFAULT_PROOF_SIZE)
    response_time: float = field(default=0.0)
    score: float = field(default=0.0)
    verification_result: bool = field(default=False)

    def __post_init__(self):
        circuit_store = CircuitStore()
        circuit = circuit_store.get_circuit(circuit_id=self.circuit_id)
        if circuit is None:
            self.maximum_response_time = CRICUIT_TIMEOUT_SECONDS
        self.maximum_response_time = circuit.timeout

    def to_dict(self) -> dict:
        """Convert the evaluation item to a dictionary for JSON serialization."""
        return {
            "circuit_id": str(self.circuit_id),
            "uid": int(self.uid),
            "minimum_response_time": float(self.minimum_response_time),
            "maximum_response_time": float(self.maximum_response_time),
            "proof_size": int(self.proof_size),
            "response_time": float(self.response_time),
            "score": float(self.score),
            "verification_result": bool(self.verification_result),
        }


class CircuitEvaluationData:
    """
    Data collected from the evaluation of the circuit.
    """

    def __init__(self, circuit: Circuit, evaluation_store_path: str):
        self.circuit = circuit
        self.store_path = evaluation_store_path
        self.data: list[CircuitEvaluationItem] = []

        os.makedirs(os.path.dirname(evaluation_store_path), exist_ok=True)

        try:
            if os.path.exists(evaluation_store_path):
                with open(evaluation_store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.data = [CircuitEvaluationItem(**item) for item in data]
        except Exception as e:
            logging.error(
                f"Failed to load evaluation data for model {self.circuit.id}, starting fresh: {e}"
            )
            self.data = []

        if not self.data:
            with open(evaluation_store_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def update(self, item: CircuitEvaluationItem):
        """Update evaluation data, maintaining size limit."""
        for i, existing_item in enumerate(self.data):
            if existing_item.uid == item.uid:
                self.data[i] = item
                break
        else:
            self.data.append(item)

        if len(self.data) > MAX_EVALUATION_ITEMS:
            self.data = self.data[-MAX_EVALUATION_ITEMS:]

        try:
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump([item.to_dict() for item in self.data], f)
        except Exception as e:
            logging.error(f"Failed to save evaluation data: {e}")

    @property
    def verification_ratio(self) -> float:
        """Get the ratio of successful verifications from recent evaluation data."""
        if not self.data:
            return 0.0

        successful = sum(1 for item in self.data if item.verification_result)
        return successful / len(self.data)

    def get_successful_response_times(self) -> list[float]:
        if not self.data:
            return []

        return sorted(
            r.response_time
            for r in self.data
            if r.verification_result and r.response_time > 0
        )

    @property
    def minimum_response_time(self) -> float:
        response_times = self.get_successful_response_times()

        if not response_times or len(response_times) in [0, 1]:
            return 0.0

        return torch.clamp(
            torch.min(torch.tensor(response_times)),
            0,
            self.circuit.timeout,
        ).item()

    @property
    def maximum_response_time(self) -> float:
        """Get maximum response time from evaluation data."""

        response_times = self.get_successful_response_times()
        if not response_times or len(response_times) in [0, 1]:
            return CRICUIT_TIMEOUT_SECONDS

        sample_size = max(int(len(response_times) * MAXIMUM_SCORE_MEDIAN_SAMPLE), 1)

        return torch.clamp(
            torch.median(torch.tensor(response_times[-sample_size:])),
            0,
            self.circuit.timeout,
        ).item()


class Circuit:
    """
    A class representing a circuit.
    """

    def __init__(self, circuit_id: str):
        """
        Initialize a Model instance.

        Args:
            circuit_id (str): Unique identifier for the model.
        """
        self.paths = CircuitPaths(circuit_id)
        self.metadata = CircuitMetadata.from_file(self.paths.metadata)
        self.id = circuit_id
        self.proof_system = ProofSystem[self.metadata.proof_system]
        self.paths.set_proof_system_paths(self.proof_system)
        self.settings = {}
        self.evaluation_data = CircuitEvaluationData(
            self, self.paths.evaluation_data
        )
        # if timeout attribute exists and is not None, else default
        self.timeout = (
            self.metadata.timeout
            if hasattr(self.metadata, 'timeout') and self.metadata.timeout is not None
            else CRICUIT_TIMEOUT_SECONDS
        )
        try:
            with open(self.paths.settings, "r", encoding="utf-8") as f:
                self.settings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logging.warning(
                f"Failed to load settings for model {self.id}. Using default settings."
            )
        self.input_handler = InputRegistry.get_handler(self.id)

    def __str__(self):
        return (
            f"{self.metadata.name} v{self.metadata.version} using {self.proof_system}"
        )
