import os
import json
import time
import shutil
import bittensor as bt
from ..models.sota import SotaState
from ..models.neuron import NeuronState


class SotaManager:
    def __init__(self, sota_directory: str):
        self.sota_directory = sota_directory
        self.sota_state_path = os.path.join(sota_directory, "sota_state.json")
        self.sota_state = self._load_state()

    def _load_state(self) -> SotaState:
        try:
            if os.path.exists(self.sota_state_path):
                with open(self.sota_state_path, "r") as f:
                    data = json.load(f)
                    return SotaState(**data)
            return SotaState()
        except Exception as e:
            bt.logging.error(f"Error loading SOTA state: {e}")
            return SotaState()

    def _save_state(self):
        try:
            with open(self.sota_state_path, "w") as f:
                json.dump(self.sota_state.__dict__, f, indent=4)
        except Exception as e:
            bt.logging.error(f"Error saving SOTA state: {e}")

    def preserve_circuit(self, circuit_dir: str, neuron_state: NeuronState):
        try:
            sota_circuit_dir = os.path.join(self.sota_directory, "circuit")
            if os.path.exists(sota_circuit_dir):
                shutil.rmtree(sota_circuit_dir)
            shutil.copytree(circuit_dir, sota_circuit_dir)

            self.sota_state = SotaState(
                sota_relative_score=neuron_state.sota_relative_score,
                hash=neuron_state.hash,
                hotkey=neuron_state.hotkey,
                proof_size=neuron_state.proof_size,
                response_time=neuron_state.response_time,
                timestamp=int(time.time()),
                raw_accuracy=neuron_state.raw_accuracy,
            )
            self._save_state()

            bt.logging.success(
                f"New SOTA achieved by {neuron_state.hotkey}! "
                f"Score: {self.sota_state.sota_relative_score:.4f}, "
                f"Raw Accuracy: {self.sota_state.raw_accuracy:.4f}, "
                f"Proof Size: {self.sota_state.proof_size:.2f} bytes, "
                f"Response Time: {self.sota_state.response_time:.4f}s"
            )
        except Exception as e:
            bt.logging.error(f"Error preserving SOTA circuit: {e}")

    def check_if_sota(
        self, sota_relative_score: float, proof_size: float, response_time: float
    ) -> bool:
        EPSILON = 1e-6

        if sota_relative_score < self.sota_state.sota_relative_score - EPSILON:
            return False
        if proof_size > self.sota_state.proof_size + EPSILON:
            return False
        if response_time > self.sota_state.response_time + EPSILON:
            return False

        metrics_equal = (
            abs(sota_relative_score - self.sota_state.sota_relative_score) < EPSILON
            and abs(proof_size - self.sota_state.proof_size) < EPSILON
            and abs(response_time - self.sota_state.response_time) < EPSILON
        )
        if metrics_equal:
            return False

        return True

    @property
    def current_state(self) -> SotaState:
        return self.sota_state
