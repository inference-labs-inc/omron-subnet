import os
import shutil
from typing import Dict, Tuple, Union, List, Optional
import bittensor as bt
import torch
import platform
import asyncio
import json
import traceback
import threading
import queue
import time
from async_substrate_interface.types import ScaleObj
from substrateinterface.utils.ss58 import ss58_encode

from .models.neuron import NeuronState
from .services.circuit_validator import CircuitValidator
from .services.circuit_manager import CircuitManager
from bittensor.core.chain_data import decode_account_id
from .services.circuit_evaluator import CircuitEvaluator
from .services.sota_manager import SotaManager

from .competition_manager import CompetitionManager
from .utils.cleanup import register_cleanup_handlers
from constants import TEMP_FOLDER
from _validator.utils.uid import get_queryable_uids
from utils.wandb_logger import safe_log


class CompetitionThread(threading.Thread):
    def __init__(
        self,
        competition: "Competition",
        pause_requests_event: threading.Event,
        config: bt.config,
    ):
        super().__init__()
        bt.logging.info("=== Competition Thread Constructor Start ===")
        self.competition = competition
        self.pause_requests_event = pause_requests_event
        self._should_run = threading.Event()
        self._should_run.set()
        self.task_queue = queue.Queue()
        self.daemon = True

        self.subtensor = bt.subtensor(config=config)

        bt.logging.info("Competition thread initialized with:")
        bt.logging.info(f"- Competition ID: {self.competition.competition_id}")
        bt.logging.info(f"- Competition Dir: {self.competition.competition_directory}")
        bt.logging.info(f"- Thread Daemon: {self.daemon}")
        bt.logging.info("=== Competition Thread Constructor End ===")

    def run(self):
        bt.logging.info("Competition thread starting...")

        while self._should_run.is_set():
            try:
                if (
                    not self.competition.download_queue
                    and not self.competition.current_download
                ):
                    time.sleep(1)
                    continue

                self._process_next_download()

            except Exception as e:
                bt.logging.error(f"Error in competition thread cycle: {str(e)}")
                bt.logging.error(f"Stack trace: {traceback.format_exc()}")
                time.sleep(5)

        bt.logging.info("Competition thread stopped")

    def _process_next_download(self):
        if not self.competition.current_download and self.competition.download_queue:
            self.competition.current_download = self.competition.download_queue.pop(0)
            uid, hotkey, hash = self.competition.current_download

            try:
                if self.competition.process_downloads_sync():
                    bt.logging.success(
                        "Circuit download successful, starting evaluation..."
                    )
                    circuit_dir = os.path.join(self.competition.temp_directory, hash)

                    self.pause_requests_event.set()
                    bt.logging.debug(
                        "Pausing main request loop for circuit evaluation..."
                    )

                    try:
                        score = self.competition._evaluate_circuit(
                            circuit_dir, hash, hotkey, self.competition.accuracy_weight
                        )
                        bt.logging.info(
                            f"Circuit evaluation complete with score {score}"
                        )

                        if score > 0 and self.competition.sota_manager.check_if_sota(
                            score,
                            self.competition.miner_states[hotkey].proof_size,
                            self.competition.miner_states[hotkey].response_time,
                        ):
                            bt.logging.success(
                                "New SOTA circuit detected, preserving..."
                            )
                            self.competition.sota_manager.preserve_circuit(
                                circuit_dir,
                                self.competition.miner_states[hotkey],
                                self.competition.miner_states,
                            )
                    except Exception as e:
                        bt.logging.error(f"Error during circuit evaluation: {str(e)}")
                        bt.logging.error(f"Stack trace: {traceback.format_exc()}")
                    finally:
                        self.competition.cleanup_circuit_dir(circuit_dir)
                        bt.logging.info("Resuming main request loop...")
                        self.pause_requests_event.clear()
                else:
                    bt.logging.error(
                        f"Circuit download or evaluation failed for {hash[:8]} from {hotkey[:8]}"
                    )
            finally:
                self.competition.clear_current_download()

    def stop(self):
        bt.logging.info("=== Competition Thread Stop Start ===")
        self._should_run.clear()
        bt.logging.info("Competition thread stop signal sent")
        bt.logging.info("=== Competition Thread Stop End ===")


class Competition:
    def __init__(
        self,
        competition_id: int,
        metagraph: bt.metagraph,
        wallet: bt.wallet,
        config: bt.config,
    ):
        bt.logging.info("=== Competition Module Initialization Start ===")
        bt.logging.info(f"Initializing competition module with ID {competition_id}...")
        self.competition_id = competition_id
        self.competition_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), str(competition_id)
        )
        self.temp_directory = os.path.join(
            TEMP_FOLDER, f"competition_{str(competition_id)}"
        )
        self.sota_directory = os.path.join(self.competition_directory, "sota")

        bt.logging.info("Creating competition directories...")
        os.makedirs(self.temp_directory, exist_ok=True)
        os.makedirs(self.sota_directory, exist_ok=True)
        bt.logging.info(
            f"Directories created: {self.temp_directory}, {self.sota_directory}"
        )

        register_cleanup_handlers()

        bt.logging.info("Loading config and initializing subtensor...")
        config_path = os.path.join(
            self.competition_directory, "competition_config.json"
        )
        with open(config_path) as f:
            self.config = json.load(f)
            self.accuracy_weight = (
                self.config.get("evaluation", {})
                .get("scoring_weights", {})
                .get("accuracy", 0.4)
            )
            bt.logging.debug(f"Loaded accuracy weight: {self.accuracy_weight}")

        self.metagraph = metagraph
        self.wallet = wallet
        self.dendrite = bt.dendrite(wallet=wallet)

        bt.logging.info("Initializing competition manager...")
        self.competition_manager = CompetitionManager(self.competition_directory)
        bt.logging.info("Loading baseline model...")
        self.baseline_model = self._load_model()
        bt.logging.info("Initializing SOTA manager...")
        self.sota_manager = SotaManager(self.sota_directory)
        self.circuit_manager = CircuitManager(
            self.temp_directory, self.competition_id, self.dendrite
        )
        self.circuit_validator = CircuitValidator()
        self.circuit_evaluator = CircuitEvaluator(
            self.baseline_model, self.competition_directory, self.sota_manager
        )

        self.miner_states: Dict[str, NeuronState] = {}

        self.download_queue: List[Tuple[int, str, str]] = []
        self.current_download: Optional[Tuple[int, str, str]] = None
        self.download_complete = asyncio.Event()
        self.download_task: Optional[asyncio.Task] = None
        self.download_lock = asyncio.Lock()

        bt.logging.info("=== Starting Competition Thread ===")
        self.pause_requests_event = threading.Event()
        bt.logging.info("Created pause_requests_event")
        self.competition_thread = CompetitionThread(
            self, self.pause_requests_event, config
        )
        bt.logging.info("Competition thread instance created")
        self.competition_thread.start()
        bt.logging.info("Competition thread started")
        bt.logging.info("=== Competition Thread Started ===")

        safe_log(
            {
                "competition_id": competition_id,
                "competition_status": "initialized",
                "competition_directory": str(self.competition_directory),
                "sota_directory": str(self.sota_directory),
            }
        )
        bt.logging.info("=== Competition Module Initialization Complete ===")

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
            bt.logging.debug(f"Using netuid: {self.metagraph.netuid}")
            safe_log(
                {
                    "competition_status": "fetching_commitments",
                    "queryable_uids_count": len(queryable_uids),
                    "netuid": self.metagraph.netuid,
                }
            )

            commitment_map = self.competition_thread.subtensor.substrate.query_map(
                module="Commitments",
                storage_function="CommitmentOf",
                params=[self.metagraph.netuid],
            )

            if not commitment_map:
                bt.logging.debug("No commitments found in query result")
                return []

            if not hasattr(commitment_map, "__iter__"):
                bt.logging.warning("Query result is not iterable")
                return []

            result_count = 0
            for result in commitment_map:
                result_count += 1
                try:
                    if not result or len(result) != 2:
                        continue

                    acc, info = result
                    if not acc:
                        continue

                    if isinstance(acc, (list, tuple)) and len(acc) > 0:
                        if isinstance(acc[0], (list, tuple)):
                            acc_bytes = bytes(acc[0])
                        else:
                            acc_bytes = bytes(acc)
                    else:
                        continue

                    try:
                        ss58_address = ss58_encode(acc_bytes)
                        hotkey = decode_account_id(acc_bytes)
                        if ss58_address != hotkey:
                            bt.logging.warning(f"SS58 address mismatch for {hotkey}")
                            continue
                    except Exception as e:
                        bt.logging.debug(f"Failed to decode account ID: {e}")
                        continue

                    if hotkey not in hotkey_to_uid:
                        continue

                    uid = hotkey_to_uid[hotkey]

                    if not isinstance(info, (dict, ScaleObj)):
                        bt.logging.debug(
                            f"No valid commitment info for {hotkey} (UID {uid})"
                        )
                        continue

                    if isinstance(info, ScaleObj):
                        info = info.value

                    commitment_info = info.get("info", {})
                    if not commitment_info or not isinstance(commitment_info, dict):
                        bt.logging.debug(
                            f"Invalid commitment info structure for {hotkey}"
                        )
                        continue

                    fields = commitment_info.get("fields", [])
                    if (
                        not fields
                        or not isinstance(fields, (list, tuple))
                        or not fields[0]
                    ):
                        bt.logging.debug(f"Invalid fields structure for {hotkey}")
                        continue

                    commitment = fields[0][0]
                    if not commitment or not isinstance(commitment, dict):
                        bt.logging.debug(f"Empty commitment for {hotkey}")
                        continue

                    commitment_type = next(iter(commitment.keys()))
                    bytes_tuple = commitment[commitment_type][0]

                    if not isinstance(bytes_tuple, (list, tuple)):
                        bt.logging.debug(f"Invalid bytes tuple for {hotkey}")
                        continue

                    try:
                        hash = bytes(bytes_tuple).decode()
                    except Exception as e:
                        bt.logging.debug(f"Failed to decode hash for {hotkey}: {e}")
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
                        safe_log(
                            {
                                "competition_status": "new_circuit_detected",
                                "miner_hotkey": hotkey,
                                "miner_uid": uid,
                                "circuit_hash": hash,
                            }
                        )

                except Exception as e:
                    bt.logging.error(f"Error processing commitment: {e}")
                    bt.logging.error(f"Stack trace: {traceback.format_exc()}")
                    safe_log(
                        {
                            "competition_status": "commitment_error",
                            "error": str(e),
                        }
                    )
                    continue

            bt.logging.debug(f"Processed {result_count} commitment results")

        except Exception as e:
            bt.logging.error(f"Error fetching commitments: {e}")
            bt.logging.error(f"Stack trace: {traceback.format_exc()}")
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
                            sota_relative_score=0.0,
                            proof_size=0,
                            response_time=0.0,
                            verification_result=False,
                            raw_accuracy=0.0,
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
        """Clear the current download without cleaning up the directory."""
        self.current_download = None

    def cleanup_circuit_dir(self, circuit_dir: str) -> None:
        """Clean up a circuit directory if it exists."""
        if os.path.exists(circuit_dir):
            bt.logging.info(f"Cleaning up circuit directory: {circuit_dir}")
            shutil.rmtree(circuit_dir)

    def process_downloads_sync(self) -> bool:
        """Process downloads synchronously."""
        try:
            current_download = self.get_current_download()
            if not current_download:
                return False

            uid, hotkey, hash = current_download
            bt.logging.info(
                f"Processing download for circuit {hash[:8]}... from {hotkey[:8]}..."
            )

            axon = self.metagraph.axons[uid]
            circuit_dir = os.path.join(self.temp_directory, hash)
            os.makedirs(circuit_dir, exist_ok=True)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                download_success = loop.run_until_complete(
                    self.circuit_manager.download_files(axon, hash, circuit_dir)
                )
            finally:
                loop.close()
                asyncio.set_event_loop(None)

            if download_success:
                bt.logging.info(f"Download completed for {hash[:8]}, validating...")

                if self.circuit_validator.validate_files(circuit_dir):
                    self.miner_states[hotkey] = NeuronState(
                        hotkey=hotkey,
                        uid=uid,
                        sota_relative_score=0.0,
                        proof_size=0,
                        response_time=0.0,
                        verification_result=False,
                        raw_accuracy=0.0,
                        hash=hash,
                    )
                    bt.logging.success(
                        f"Successfully downloaded and validated circuit {hash[:8]}..."
                    )
                    return True
                else:
                    bt.logging.warning(f"Circuit validation failed for {hash[:8]}...")
            else:
                bt.logging.warning(f"Failed to download circuit {hash[:8]}...")

            self.cleanup_circuit_dir(circuit_dir)
            return False

        except Exception as e:
            bt.logging.error(f"Error processing download: {e}")
            traceback.print_exc()
            if current_download:
                circuit_dir = os.path.join(self.temp_directory, current_download[2])
                self.cleanup_circuit_dir(circuit_dir)
            return False
        finally:
            self.clear_current_download()

    def run_single_evaluation(self) -> bool:
        if not self.pending_evaluations:
            return False

        axon, circuit_dir = self.pending_evaluations.pop(0)
        try:
            self._evaluate_circuit(
                circuit_dir, axon.hotkey, axon.hotkey, self.accuracy_weight
            )
            return True
        finally:
            self.circuit_manager.cleanup_temp_files(circuit_dir)

    def _evaluate_circuit(
        self,
        circuit_dir: str,
        circuit_uid: str,
        circuit_owner: str,
        accuracy_weight: float,
    ) -> float:
        try:
            sota_relative_score, proof_size, response_time, success = (
                self.circuit_evaluator.evaluate(circuit_dir, accuracy_weight)
            )

            if not success:
                return 0.0

            if circuit_owner in self.miner_states:
                self.miner_states[circuit_owner].sota_relative_score = (
                    sota_relative_score
                )
                self.miner_states[circuit_owner].proof_size = proof_size
                self.miner_states[circuit_owner].response_time = response_time
                self.miner_states[circuit_owner].verification_result = True

            safe_log(
                {
                    "circuit_eval_status": "eval_success",
                    "circuit_uid": circuit_uid,
                    "circuit_owner": circuit_owner,
                    "sota_relative_score": float(sota_relative_score),
                    "proof_size": -1 if proof_size == float("inf") else int(proof_size),
                    "response_time": (
                        -1 if response_time == float("inf") else float(response_time)
                    ),
                }
            )

            self._update_competition_metrics()

            return sota_relative_score
        except Exception as e:
            bt.logging.error(f"Error in competition thread cycle: {str(e)}")
            bt.logging.error(f"Stack trace: {traceback.format_exc()}")
            return 0.0

    def _update_competition_metrics(self):
        self.competition_manager.increment_circuits_evaluated()

        active_miners = [
            (k, v)
            for k, v in self.miner_states.items()
            if v.verification_result and v.sota_relative_score > 0
        ]
        active_participants = len(active_miners)
        self.competition_manager.update_active_participants(active_participants)

        sota_state = self.sota_manager.current_state

        if active_miners:
            sorted_by_accuracy = sorted(
                active_miners, key=lambda x: x[1].raw_accuracy, reverse=True
            )
            sorted_by_proof_size = sorted(active_miners, key=lambda x: x[1].proof_size)
            sorted_by_response_time = sorted(
                active_miners, key=lambda x: x[1].response_time
            )

            metrics = {}

            for miner_hotkey, miner_state in active_miners:
                accuracy_rank = (
                    next(
                        i
                        for i, (h, _) in enumerate(sorted_by_accuracy)
                        if h == miner_hotkey
                    )
                    + 1
                )
                proof_size_rank = (
                    next(
                        i
                        for i, (h, _) in enumerate(sorted_by_proof_size)
                        if h == miner_hotkey
                    )
                    + 1
                )
                response_time_rank = (
                    next(
                        i
                        for i, (h, _) in enumerate(sorted_by_response_time)
                        if h == miner_hotkey
                    )
                    + 1
                )
                overall_rank = (
                    accuracy_rank + proof_size_rank + response_time_rank
                ) / 3

                metrics.update(
                    {
                        f"hotkey.{miner_hotkey}.sota_score": miner_state.sota_relative_score,
                        f"hotkey.{miner_hotkey}.raw_accuracy": miner_state.raw_accuracy,
                        f"hotkey.{miner_hotkey}.proof_size": miner_state.proof_size,
                        f"hotkey.{miner_hotkey}.response_time": miner_state.response_time,
                        f"hotkey.{miner_hotkey}.relative_to_sota.accuracy": (
                            miner_state.raw_accuracy / sota_state.raw_accuracy
                            if sota_state.raw_accuracy > 0
                            else 0
                        ),
                        f"hotkey.{miner_hotkey}.relative_to_sota.proof_size": (
                            sota_state.proof_size / miner_state.proof_size
                            if miner_state.proof_size > 0
                            else 0
                        ),
                        f"hotkey.{miner_hotkey}.relative_to_sota.response_time": (
                            sota_state.response_time / miner_state.response_time
                            if miner_state.response_time > 0
                            else 0
                        ),
                        f"hotkey.{miner_hotkey}.rank.overall": overall_rank,
                        f"hotkey.{miner_hotkey}.rank.accuracy": accuracy_rank,
                        f"hotkey.{miner_hotkey}.rank.proof_size": proof_size_rank,
                        f"hotkey.{miner_hotkey}.rank.response_time": response_time_rank,
                    }
                )

                if hasattr(miner_state, "last_score"):
                    time_delta = (
                        time.time() - miner_state.last_score_time
                        if hasattr(miner_state, "last_score_time")
                        else 1
                    )
                    improvement_rate = (
                        miner_state.sota_relative_score - miner_state.last_score
                    ) / time_delta
                    metrics.update(
                        {
                            f"hotkey.{miner_hotkey}.historical.improvement_rate": improvement_rate,
                            f"hotkey.{miner_hotkey}.historical.best_sota_score": max(
                                miner_state.sota_relative_score,
                                getattr(miner_state, "best_sota_score", 0),
                            ),
                        }
                    )

                miner_state.last_score = miner_state.sota_relative_score
                miner_state.last_score_time = time.time()
                miner_state.best_sota_score = max(
                    miner_state.sota_relative_score,
                    getattr(miner_state, "best_sota_score", 0),
                )

            metrics.update(
                {
                    "active_participants": active_participants,
                    "avg_raw_accuracy": sum(m[1].raw_accuracy for m in active_miners)
                    / active_participants,
                    "avg_proof_size": sum(m[1].proof_size for m in active_miners)
                    / active_participants,
                    "avg_response_time": sum(m[1].response_time for m in active_miners)
                    / active_participants,
                    "sota_relative_score": sota_state.sota_relative_score,
                    "sota_hotkey": sota_state.hotkey,
                    "sota_proof_size": sota_state.proof_size,
                    "sota_response_time": sota_state.response_time,
                }
            )

            self.competition_manager.log_metrics(metrics)
