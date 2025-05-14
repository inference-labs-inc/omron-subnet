import os
import json
import time
import shutil
import bittensor as bt
from ..models.sota import SotaState
from ..models.neuron import NeuronState
import traceback
from _validator.utils.logging import log_sota_scores


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
        hotkey: str,
        uid: int,
        raw_accuracy: float,
        proof_size: float,
        response_time: float,
        hash: str,
        miner_states: dict[str, NeuronState],
    ):
        """Preserves the circuit if it represents a new SOTA based on raw performance."""
        try:
            is_new_sota = False
            if self.sota_state.raw_accuracy == 0 or self.sota_state.hash is None:
                is_new_sota = True
            elif raw_accuracy > self.sota_state.raw_accuracy * 1.02:
                is_new_sota = True
            elif raw_accuracy == self.sota_state.raw_accuracy:
                if proof_size < self.sota_state.proof_size:
                    is_new_sota = True
                elif proof_size == self.sota_state.proof_size:
                    if response_time < self.sota_state.response_time:
                        is_new_sota = True

            if is_new_sota:
                sota_circuit_dir = os.path.join(self.sota_directory, "circuit")
                if os.path.exists(sota_circuit_dir):
                    shutil.rmtree(sota_circuit_dir)
                shutil.copytree(circuit_dir, sota_circuit_dir)

                self.sota_state = SotaState(
                    sota_relative_score=1.0,
                    hash=hash,
                    hotkey=hotkey,
                    uid=uid,
                    proof_size=proof_size,
                    response_time=response_time,
                    timestamp=int(time.time()),
                    raw_accuracy=raw_accuracy,
                )
                self._save_state()

                bt.logging.success(
                    f"New SOTA achieved by {hotkey}! "
                    f"Raw Accuracy: {raw_accuracy:.4f}, "
                    f"Proof Size: {proof_size:.0f} bytes, "
                    f"Response Time: {response_time:.4f}s"
                )

            else:
                bt.logging.debug(f"Submission from {hotkey} did not improve SOTA.")

        except Exception as e:
            bt.logging.error(f"Error preserving SOTA circuit: {e}")
            bt.logging.error(traceback.format_exc())
        finally:
            self.recalculate_miner_scores(miner_states)

    def recalculate_miner_scores(self, miner_states: dict[str, NeuronState]) -> None:
        """Recalculate all miner scores using a strict ranking system."""
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

            valid_miners = {
                hotkey: state
                for hotkey, state in miner_states.items()
                if state.verification_result and state.raw_accuracy > 0
            }

            if not valid_miners:
                bt.logging.info("No valid miners to rank")
                return

            performance_scores = []

            for hotkey, state in valid_miners.items():
                accuracy_score = state.raw_accuracy / max(
                    self.sota_state.raw_accuracy,
                    max(s.raw_accuracy for s in valid_miners.values()),
                )

                min_proof_size = min(s.proof_size for s in valid_miners.values())
                proof_size_score = (
                    min_proof_size / state.proof_size if state.proof_size > 0 else 0
                )

                min_response_time = min(s.response_time for s in valid_miners.values())
                response_time_score = (
                    min_response_time / state.response_time
                    if state.response_time > 0
                    else 0
                )

                total_score = (
                    accuracy_score * weights["accuracy"]
                    + proof_size_score * weights["proof_size"]
                    + response_time_score * weights["response_time"]
                )

                score = total_score
                performance_scores.append((hotkey, score))

            performance_scores.sort(key=lambda x: x[1], reverse=True)
            max_score = max(score for _, score in performance_scores)

            for hotkey, score in performance_scores:
                normalized_score = score / max_score if max_score > 0 else 0
                miner_states[hotkey].sota_relative_score = normalized_score

            for hotkey, state in miner_states.items():
                if hotkey not in [h for h, _ in performance_scores]:
                    state.sota_relative_score = 0.0

            log_sota_scores(performance_scores, miner_states, max_score)

        except Exception as e:
            bt.logging.error(f"Error recalculating miner scores: {e}")
            bt.logging.error(traceback.format_exc())

    @property
    def current_state(self) -> SotaState:
        return self.sota_state
