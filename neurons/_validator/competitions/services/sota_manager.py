import os
import json
import time
import shutil
import bittensor as bt
from ..models.sota import SotaState
from ..models.neuron import NeuronState
import torch


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

    def preserve_circuit(
        self,
        circuit_dir: str,
        neuron_state: NeuronState,
        miner_states: dict[str, NeuronState],
    ):
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
                f"Score: {neuron_state.sota_relative_score:.4f}, "
                f"Raw Accuracy: {neuron_state.raw_accuracy:.4f}, "
                f"Proof Size: {neuron_state.proof_size:.2f} bytes, "
                f"Response Time: {neuron_state.response_time:.4f}s"
            )

            self.recalculate_miner_scores(miner_states)
        except Exception as e:
            bt.logging.error(f"Error preserving SOTA circuit: {e}")

    def recalculate_miner_scores(self, miner_states: dict[str, NeuronState]) -> None:
        """Recalculate all miner scores relative to the new SOTA."""
        try:
            with open(
                os.path.join(
                    os.path.dirname(self.sota_directory), "competition_config.json"
                )
            ) as f:
                config = json.load(f)
                weights = config["evaluation"]["scoring_weights"]
        except Exception as e:
            bt.logging.error(
                f"Error loading scoring weights for recalculation, using defaults: {e}"
            )
            weights = {"accuracy": 0.4, "proof_size": 0.3, "response_time": 0.3}

        try:
            for hotkey, state in miner_states.items():
                if not state.verification_result or state.raw_accuracy == 0:
                    state.sota_relative_score = 0.0
                    continue

                accuracy_diff = max(
                    0, self.sota_state.raw_accuracy - state.raw_accuracy
                )

                if self.sota_state.proof_size > 0:
                    proof_size_diff = max(
                        0,
                        (state.proof_size - self.sota_state.proof_size)
                        / self.sota_state.proof_size,
                    )
                else:
                    proof_size_diff = 0 if state.proof_size == 0 else 1

                if self.sota_state.response_time > 0:
                    response_time_diff = max(
                        0,
                        (state.response_time - self.sota_state.response_time)
                        / self.sota_state.response_time,
                    )
                else:
                    response_time_diff = 0 if state.response_time == 0 else 1

                total_diff = torch.tensor(
                    accuracy_diff * weights["accuracy"]
                    + proof_size_diff * weights["proof_size"]
                    + response_time_diff * weights["response_time"]
                )

                state.sota_relative_score = torch.exp(-total_diff).item()
        except Exception as e:
            bt.logging.error(f"Error recalculating miner scores: {e}")

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
