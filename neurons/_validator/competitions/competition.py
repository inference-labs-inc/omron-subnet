import os
import shutil
from typing import Dict, Tuple, Union, List, Optional
import bittensor as bt
import torch
import platform
import asyncio
import json
import traceback

from .models.neuron import NeuronState
from .services.circuit_validator import CircuitValidator
from .services.circuit_manager import CircuitManager
from .services.circuit_evaluator import CircuitEvaluator
from .services.sota_manager import SotaManager
from .services.data_source import (
    CompetitionDataSource,
    RandomDataSource,
    RemoteDataSource,
    CompetitionDataProcessor,
)
from .competition_manager import CompetitionManager
from .utils.cleanup import register_cleanup_handlers
from constants import TEMP_FOLDER, MAINNET_TESTNET_UIDS
from _validator.utils.uid import get_queryable_uids
from scalecodec.utils.ss58 import ss58_encode
from utils.wandb_logger import safe_log


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
        self.circuit_manager = None
        self.circuit_validator = CircuitValidator()
        self.circuit_evaluator = CircuitEvaluator(
            self.baseline_model, self.competition_directory, self.sota_manager
        )

        self.metagraph = metagraph
        self.subtensor = subtensor
        self.miner_states: Dict[str, NeuronState] = {}

        self.download_queue: List[Tuple[int, str, str]] = []
        self.current_download: Optional[Tuple[int, str, str]] = None
        self.download_complete = asyncio.Event()
        self.download_task: Optional[asyncio.Task] = None
        self.download_lock = asyncio.Lock()

        safe_log(
            {
                "competition_id": competition_id,
                "competition_status": "initialized",
                "competition_directory": str(self.competition_directory),
                "sota_directory": str(self.sota_directory),
            }
        )

    def _setup_data_source(self) -> CompetitionDataSource:
        try:
            config_path = os.path.join(
                self.competition_directory, "competition_config.json"
            )
            with open(config_path) as f:
                config = json.load(f)
                data_config = config.get("data_source", {})

                processor = None
                processor_path = os.path.join(
                    self.competition_directory, "data_processor.py"
                )
                if os.path.exists(processor_path):
                    import importlib.util

                    spec = importlib.util.spec_from_file_location(
                        "data_processor", processor_path
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, CompetitionDataProcessor)
                            and attr != CompetitionDataProcessor
                        ):
                            processor = attr()
                            break

                if data_config.get("type") == "remote":
                    data_source = RemoteDataSource(
                        self.competition_directory, processor
                    )
                    if not data_source.sync_data():
                        bt.logging.error("Failed to sync remote data source")
                        return RandomDataSource(self.competition_directory, processor)
                    return data_source
                return RandomDataSource(self.competition_directory, processor)

        except Exception as e:
            bt.logging.error(f"Error setting up data source: {e}")
            traceback.print_exc()
            return RandomDataSource(self.competition_directory)

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
                "Competitions are only supported on macOS arm64 architecture."
            )
            bt.logging.critical(
                "To remain in consensus, please use a supported platform."
            )
            bt.logging.critical(
                "While the validator will continue to run, it will not be able to "
                "correctly evaluate competitions."
            )

        hotkeys = self.metagraph.hotkeys
        self.miner_states = {k: v for k, v in self.miner_states.items() if k in hotkeys}

        commitments = []
        queryable_uids = list(get_queryable_uids(self.metagraph))
        hotkey_to_uid = {self.metagraph.hotkeys[uid]: uid for uid in queryable_uids}

        try:
            bt.logging.debug(
                f"Fetching commitments for {len(queryable_uids)} queryable UIDs"
            )
            safe_log(
                {
                    "competition_status": "fetching_commitments",
                    "queryable_uids_count": len(queryable_uids),
                }
            )

            commitment_map = self.subtensor.substrate.query_map(
                module="Commitments",
                storage_function="CommitmentOf",
                params=[self.metagraph.netuid],
            )

            for acc, info in commitment_map:
                try:
                    if self.metagraph.netuid == next(
                        (
                            testnet
                            for mainnet, testnet in MAINNET_TESTNET_UIDS
                            if mainnet == 2
                        ),
                        None,
                    ):
                        if not acc or not acc[0]:
                            continue
                        acc = ss58_encode(bytes(acc[0]))

                    if acc not in hotkey_to_uid:
                        continue

                    uid = hotkey_to_uid[acc]

                    if not info or not isinstance(info, dict):
                        bt.logging.debug(
                            f"No valid commitment info for {acc} (UID {uid})"
                        )
                        continue

                    commitment_info = info.get("info", {})
                    if not commitment_info or not isinstance(commitment_info, dict):
                        bt.logging.debug(f"Invalid commitment info structure for {acc}")
                        continue

                    fields = commitment_info.get("fields", [])
                    if not fields or not isinstance(fields, list) or not fields[0]:
                        bt.logging.debug(f"Invalid fields structure for {acc}")
                        continue

                    raw64_data = fields[0][0].get("Raw64") if fields[0][0] else None
                    if not raw64_data:
                        bt.logging.debug(f"No Raw64 data for {acc}")
                        continue

                    if self.metagraph.netuid == next(
                        (
                            testnet
                            for mainnet, testnet in MAINNET_TESTNET_UIDS
                            if mainnet == 2
                        ),
                        None,
                    ):
                        if not isinstance(raw64_data, list) or not raw64_data[0]:
                            continue
                        hash = bytes(raw64_data[0]).decode("utf-8")
                    else:
                        if not isinstance(raw64_data, str) or not raw64_data.startswith(
                            "0x"
                        ):
                            continue
                        hash = bytes.fromhex(raw64_data[2:]).decode("utf-8")

                    if (
                        acc not in self.miner_states
                        or self.miner_states[acc].hash != hash
                    ):
                        if hash in {state.hash for state in self.miner_states.values()}:
                            bt.logging.warning(
                                f"Circuit with hash {hash} already exists for another miner"
                            )
                            continue
                        commitments.append((uid, acc, hash))
                        bt.logging.success(
                            f"New circuit detected for {acc} with hash {hash}"
                        )
                        safe_log(
                            {
                                "competition_status": "new_circuit_detected",
                                "miner_hotkey": acc,
                                "miner_uid": uid,
                                "circuit_hash": hash,
                            }
                        )

                except Exception as e:
                    bt.logging.error(f"Error processing commitment for {acc}: {e}")
                    traceback.print_exc()
                    safe_log(
                        {
                            "competition_status": "commitment_error",
                            "miner_hotkey": acc,
                            "error": str(e),
                        }
                    )
                    continue

        except Exception as e:
            bt.logging.error(f"Error fetching commitments: {e}")
            traceback.print_exc()
            safe_log(
                {
                    "competition_status": "fetch_error",
                    "error": str(e),
                }
            )

        safe_log(
            {
                "competition_status": "fetch_complete",
                "total_commitments": len(commitments),
            }
        )
        return commitments

    async def process_downloads(self) -> bool:
        try:
            async with self.download_lock:
                if not self.download_queue and not self.current_download:
                    return False

                if not self.current_download and self.download_queue:
                    self.current_download = self.download_queue.pop(0)

            if self.current_download:
                uid, hotkey, hash = self.current_download
                safe_log(
                    {
                        "competition_status": "downloading_circuit",
                        "miner_hotkey": hotkey,
                        "miner_uid": uid,
                        "circuit_hash": hash,
                    }
                )

                axon = self.metagraph.axons[uid]
                circuit_dir = os.path.join(self.temp_directory, hash)
                os.makedirs(circuit_dir, exist_ok=True)

                if await self.circuit_manager.download_files(axon, hash, circuit_dir):
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
                        safe_log(
                            {
                                "competition_status": "download_success",
                                "miner_hotkey": hotkey,
                                "circuit_hash": hash,
                            }
                        )
                        return True
                    else:
                        safe_log(
                            {
                                "competition_status": "validation_failed",
                                "miner_hotkey": hotkey,
                                "circuit_hash": hash,
                            }
                        )

                else:
                    safe_log(
                        {
                            "competition_status": "download_failed",
                            "miner_hotkey": hotkey,
                            "circuit_hash": hash,
                        }
                    )

                if os.path.exists(circuit_dir):
                    shutil.rmtree(circuit_dir)

                self.current_download = None
                return False

        except Exception as e:
            bt.logging.error(f"Error in download processing: {e}")
            traceback.print_exc()
            if self.current_download:
                uid, hotkey, hash = self.current_download
                safe_log(
                    {
                        "competition_status": "download_error",
                        "miner_hotkey": hotkey,
                        "circuit_hash": hash,
                        "error": str(e),
                    }
                )
                circuit_dir = os.path.join(self.temp_directory, hash)
                if os.path.exists(circuit_dir):
                    shutil.rmtree(circuit_dir)
                self.current_download = None
            return False

    def queue_download(self, uid: int, hotkey: str, hash: str) -> None:
        self.download_queue.append((uid, hotkey, hash))
        self.download_complete.set()

    def get_current_download(self) -> Optional[Tuple[int, str, str]]:
        return self.current_download

    def clear_current_download(self) -> None:
        if self.current_download:
            circuit_dir = os.path.join(self.temp_directory, self.current_download[2])
            if os.path.exists(circuit_dir):
                shutil.rmtree(circuit_dir)
            self.current_download = None

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
        safe_log(
            {
                "competition_status": "evaluating_circuit",
                "miner_hotkey": miner_axon.hotkey,
            }
        )

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

                safe_log(
                    {
                        "competition_status": "evaluation_complete",
                        "miner_hotkey": hotkey,
                        "miner_uid": uid,
                        "score": float(score),
                        "proof_size": int(proof_size),
                        "response_time": float(response_time),
                        "verification_success": bool(verification_success),
                    }
                )

                if (
                    self.competition_manager.is_competition_active()
                    and self.sota_manager.check_if_sota(
                        score, proof_size, response_time
                    )
                ):
                    safe_log(
                        {
                            "competition_status": "new_sota",
                            "miner_hotkey": hotkey,
                            "miner_uid": uid,
                            "score": float(score),
                            "proof_size": int(proof_size),
                            "response_time": float(response_time),
                        }
                    )
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

    def initialize_circuit_manager(self, dendrite: bt.dendrite):
        self.circuit_manager = CircuitManager(
            self.temp_directory, self.competition_id, dendrite
        )
