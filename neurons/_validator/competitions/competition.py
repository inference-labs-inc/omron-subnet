import os
import shutil
from typing import Dict, Generator, Tuple, Union
import bittensor as bt
import torch

from .models.neuron import NeuronState
from .services.circuit_validator import CircuitValidator
from .services.circuit_manager import CircuitManager
from .services.circuit_evaluator import CircuitEvaluator
from .services.sota_manager import SotaManager
from .competition_manager import CompetitionManager
from .utils.cleanup import register_cleanup_handlers
from constants import TEMP_FOLDER


class Competition:
    def __init__(
        self, competition_id: int, metagraph: bt.metagraph, subtensor: bt.subtensor
    ):
        self.competition_id = competition_id
        self.competition_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), str(competition_id)
        )
        self.temp_directory = os.path.join(
            TEMP_FOLDER, f"competition_{str(competition_id)}"
        )
        self.sota_directory = os.path.join(self.competition_directory, "sota")

        os.makedirs(self.temp_directory, exist_ok=True)
        os.makedirs(self.sota_directory, exist_ok=True)

        register_cleanup_handlers()

        self.competition_manager = CompetitionManager(self.competition_directory)
        self.baseline_model = self._load_model()
        self.sota_manager = SotaManager(self.sota_directory)
        self.circuit_manager = CircuitManager(self.temp_directory, self.competition_id)
        self.circuit_validator = CircuitValidator()
        self.circuit_evaluator = CircuitEvaluator(self.baseline_model)

        self.metagraph = metagraph
        self.subtensor = subtensor
        self.miner_states: Dict[str, NeuronState] = {}

    def _load_model(self) -> Union[torch.nn.Module, str]:
        if not self.competition_manager.current_competition:
            raise ValueError("No competition configured")

        model_path = self.competition_manager.current_competition.baseline_model_path
        if model_path.endswith(".pt"):
            return torch.load(model_path)
        elif model_path.endswith(".onnx"):
            return model_path
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

    def sync_and_eval(self):
        self.competition_manager.update_competition_state()

        if not self.competition_manager.is_competition_active():
            bt.logging.info("Competition not active, skipping circuit sync")
            return

        for axon, circuit_dir in self._sync_circuits():
            try:
                if self.circuit_validator.validate_files(circuit_dir):
                    self._evaluate_circuit(axon, circuit_dir)
                else:
                    bt.logging.error(f"Circuit validation failed for {axon}")
                    if os.path.exists(circuit_dir):
                        shutil.rmtree(circuit_dir)
            finally:
                self.circuit_manager.cleanup_temp_files(circuit_dir)

    def _sync_circuits(self) -> Generator[Tuple[bt.axon, str]]:
        hotkeys = self.metagraph.hotkeys
        self.miner_states = {k: v for k, v in self.miner_states.items() if k in hotkeys}

        for hotkey in [
            h
            for h in hotkeys
            if not self.metagraph.validator_permit[self.metagraph.hotkeys.index(h)]
        ]:
            try:
                hash = self.subtensor.get_commitment(
                    self.metagraph.netuid, self.metagraph.hotkeys.index(hotkey)
                )
                if not hash:
                    continue

                if (
                    hotkey not in self.miner_states
                    or self.miner_states[hotkey].hash != hash
                ):
                    bt.logging.success(
                        f"New circuit detected for {hotkey} with hash {hash}"
                    )
                    axon = self.metagraph.axons[self.metagraph.hotkeys.index(hotkey)]

                    circuit_dir = os.path.join(self.temp_directory, hash)
                    os.makedirs(circuit_dir, exist_ok=True)

                    if self.circuit_manager.download_files(axon, hash, circuit_dir):
                        self.miner_states[hotkey] = NeuronState(
                            hotkey=hotkey,
                            uid=self.metagraph.hotkeys.index(hotkey),
                            score=0.0,
                            proof_size=0,
                            response_time=0.0,
                            verification_result=False,
                            accuracy=0.0,
                            hash=hash,
                        )
                        yield (axon, circuit_dir)
                    else:
                        bt.logging.error(
                            f"Failed to download circuit files for {hotkey}"
                        )
                        if os.path.exists(circuit_dir):
                            shutil.rmtree(circuit_dir)

            except Exception as e:
                bt.logging.error(f"Error getting commitment for {hotkey}: {e}")
                continue

    def _evaluate_circuit(self, miner_axon: bt.axon, circuit_dir: str) -> float:
        bt.logging.info(f"Evaluating circuit for {miner_axon}")

        accuracy_weight = self.competition_manager.get_accuracy_weight()
        bt.logging.info(f"Current accuracy weight: {accuracy_weight}")

        score, proof_size, response_time, verification_success = (
            self.circuit_evaluator.evaluate(circuit_dir, accuracy_weight)
        )

        hotkey = next(
            k for k, v in self.miner_states.items() if v.uid == miner_axon.uid
        )
        if hotkey:
            self.miner_states[hotkey].score = score
            self.miner_states[hotkey].proof_size = proof_size
            self.miner_states[hotkey].response_time = response_time
            self.miner_states[hotkey].verification_result = verification_success
            self.miner_states[hotkey].accuracy = score

            if (
                self.competition_manager.is_competition_active()
                and self.sota_manager.check_if_sota(score, proof_size, response_time)
            ):
                self.sota_manager.preserve_circuit(
                    circuit_dir, self.miner_states[hotkey]
                )

            self._update_competition_metrics(hotkey)

        status = self.competition_manager.get_competition_status()
        bt.logging.info(f"Competition status: {status}")

        bt.logging.info(f"Average accuracy score: {score}")
        bt.logging.info(f"Average proof size: {proof_size} bytes")
        bt.logging.info(f"Average response time: {response_time}s")
        bt.logging.info(
            f"Verification success rate: {1.0 if verification_success else 0.0}"
        )

        return score

    def _update_competition_metrics(self, hotkey: str):
        self.competition_manager.increment_circuits_evaluated()

        active_participants = len(
            [
                k
                for k, v in self.miner_states.items()
                if v.verification_result and v.score > 0
            ]
        )
        self.competition_manager.update_active_participants(active_participants)

        active_miners = [
            v
            for v in self.miner_states.values()
            if v.verification_result and v.score > 0
        ]
        sota_state = self.sota_manager.current_state

        metrics = {
            "active_participants": active_participants,
            "avg_accuracy": (
                sum(m.accuracy for m in active_miners) / len(active_miners)
                if active_miners
                else 0
            ),
            "avg_proof_size": (
                sum(m.proof_size for m in active_miners) / len(active_miners)
                if active_miners
                else 0
            ),
            "avg_response_time": (
                sum(m.response_time for m in active_miners) / len(active_miners)
                if active_miners
                else 0
            ),
            "sota_score": sota_state.score,
            "sota_hotkey": sota_state.hotkey,
            "sota_proof_size": sota_state.proof_size,
            "sota_response_time": sota_state.response_time,
        }
        self.competition_manager.log_metrics(metrics)
