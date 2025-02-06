import os
import shutil
from typing import Dict, Tuple, Union, List
import bittensor as bt
import torch
import platform

from .models.neuron import NeuronState
from .services.circuit_validator import CircuitValidator
from .services.circuit_manager import CircuitManager
from .services.circuit_evaluator import CircuitEvaluator
from .services.sota_manager import SotaManager
from .competition_manager import CompetitionManager
from .utils.cleanup import register_cleanup_handlers
from constants import TEMP_FOLDER, MAINNET_TESTNET_UIDS
from _validator.utils.uid import get_queryable_uids
from scalecodec.utils.ss58 import ss58_encode


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
        self.circuit_evaluator = CircuitEvaluator(
            self.baseline_model, self.competition_directory, self.sota_manager
        )

        self.metagraph = metagraph
        self.subtensor = subtensor
        self.miner_states: Dict[str, NeuronState] = {}
        self.pending_evaluations: List[Tuple[bt.axon, str]] = []

    def _load_model(self) -> Union[torch.nn.Module, str, None]:
        if not self.competition_manager.current_competition:
            bt.logging.info("No competition currently configured")
            return None

        model_path = self.competition_manager.current_competition.baseline_model_path
        model_path = os.path.join(
            self.competition_directory, os.path.basename(model_path)
        )
        bt.logging.info(f"Loading model from: {model_path}")

        if model_path.endswith(".pt"):
            return torch.load(model_path)
        elif model_path.endswith(".onnx"):
            return model_path
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

    def fetch_commitments(self) -> List[Tuple[int, str, str]]:
        if platform.system() != "Darwin" and platform.machine() != "arm64":
            bt.logging.critical(
                "Competitions are only supported on macOS arm64 architecture\n"
                "To remain in consensus, please use a supported platform.\n"
                "While the validator will continue to run, it will not be able to "
                "correctly evaluate competitions."
            )
            return []

        hotkeys = self.metagraph.hotkeys
        self.miner_states = {k: v for k, v in self.miner_states.items() if k in hotkeys}

        commitments = []
        queryable_uids = get_queryable_uids(self.metagraph)

        commitment_map = self.subtensor.substrate.query_map(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[self.metagraph.netuid],
        )

        for uid in queryable_uids:
            hotkey = self.metagraph.hotkeys[uid]
            try:
                commitment_info = None
                for acc, info in commitment_map:

                    if self.metagraph.netuid == next(
                        testnet
                        for mainnet, testnet in MAINNET_TESTNET_UIDS
                        if mainnet == 2
                    ):
                        acc = ss58_encode(bytes(acc[0]))
                        info = bytes(
                            info["info"]["fields"][0][0].get("Raw64")[0]
                        ).decode()

                    if acc == hotkey:
                        commitment_info = info
                        break

                if not commitment_info or "info" not in commitment_info:
                    bt.logging.warning(
                        f"No valid commitment found for {hotkey} (UID {uid})"
                    )
                    continue

                try:
                    raw64_field = commitment_info["info"]["fields"][0].get("Raw64")
                    if not raw64_field:
                        bt.logging.warning(f"Invalid commitment format for {hotkey}")
                        continue

                    hash = bytes.fromhex(raw64_field[2:]).decode("utf-8")
                except (KeyError, IndexError, ValueError) as e:
                    bt.logging.warning(f"Failed to parse commitment for {hotkey}: {e}")
                    continue

                if (
                    hotkey not in self.miner_states
                    or self.miner_states[hotkey].hash != hash
                ):
                    if hash in {state.hash for state in self.miner_states.values()}:
                        bt.logging.warning(
                            f"Circuit with hash {hash} already exists for another miner"
                        )
                        continue
                    commitments.append((uid, hotkey, hash))
                    bt.logging.success(
                        f"New circuit detected for {hotkey} with hash {hash}"
                    )

            except Exception as e:
                bt.logging.error(f"Error getting commitment for {hotkey}: {e}")

        return commitments

    def prepare_evaluation(self, uid: int, hotkey: str, hash: str) -> bool:
        try:
            axon = self.metagraph.axons[uid]
            circuit_dir = os.path.join(self.temp_directory, hash)
            os.makedirs(circuit_dir, exist_ok=True)

            if self.circuit_manager.download_files(axon, hash, circuit_dir):
                if self.circuit_validator.validate_files(circuit_dir):
                    self.miner_states[hotkey] = NeuronState(
                        hotkey=hotkey,
                        uid=uid,
                        score=0.0,
                        proof_size=0,
                        response_time=0.0,
                        verification_result=False,
                        accuracy=0.0,
                        hash=hash,
                    )
                    self.pending_evaluations.append((axon, circuit_dir))
                    return True
                else:
                    bt.logging.error(f"Circuit validation failed for {hotkey}")
                    if os.path.exists(circuit_dir):
                        shutil.rmtree(circuit_dir)
            else:
                bt.logging.error(f"Failed to download circuit files for {hotkey}")
                if os.path.exists(circuit_dir):
                    shutil.rmtree(circuit_dir)
        except Exception as e:
            bt.logging.error(f"Error preparing evaluation for {hotkey}: {e}")
            if os.path.exists(circuit_dir):
                shutil.rmtree(circuit_dir)
        return False

    def run_single_evaluation(self) -> bool:
        if not self.pending_evaluations:
            return False

        axon, circuit_dir = self.pending_evaluations.pop(0)
        try:
            self._evaluate_circuit(axon, circuit_dir)
            return True
        finally:
            self.circuit_manager.cleanup_temp_files(circuit_dir)

    def _evaluate_circuit(self, miner_axon: bt.axon, circuit_dir: str) -> float:
        bt.logging.info(f"Evaluating circuit for {miner_axon.hotkey}")

        accuracy_weight = self.competition_manager.get_accuracy_weight()

        score, proof_size, response_time, verification_success = (
            self.circuit_evaluator.evaluate(circuit_dir, accuracy_weight)
        )

        uid = next(
            (
                i
                for i, axon in enumerate(self.metagraph.axons)
                if axon.hotkey == miner_axon.hotkey
            ),
            None,
        )
        if uid is not None:
            hotkey = self.metagraph.hotkeys[uid]
            if hotkey:
                self.miner_states[hotkey].score = score
                self.miner_states[hotkey].proof_size = proof_size
                self.miner_states[hotkey].response_time = response_time
                self.miner_states[hotkey].verification_result = verification_success
                self.miner_states[hotkey].accuracy = score

                if (
                    self.competition_manager.is_competition_active()
                    and self.sota_manager.check_if_sota(
                        score, proof_size, response_time
                    )
                ):
                    self.sota_manager.preserve_circuit(
                        circuit_dir, self.miner_states[hotkey]
                    )

                self._update_competition_metrics(hotkey)

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
