from __future__ import annotations

import asyncio
import random
import sys
import time
import traceback
from typing import NoReturn

import bittensor as bt

from _validator.config import ValidatorConfig
from _validator.api import ValidatorAPI
from _validator.core.prometheus import (
    log_validation_time,
    start_prometheus_logging,
    stop_prometheus_logging,
)
from _validator.core.request import Request
from _validator.core.request_pipeline import RequestPipeline
from _validator.core.response_processor import ResponseProcessor
from _validator.models.miner_response import MinerResponse
from _validator.scoring.score_manager import ScoreManager
from _validator.scoring.weights import WeightsManager
from _validator.utils.api import hash_inputs
from _validator.utils.axon import query_axons
from _validator.models.request_type import RequestType
from _validator.utils.proof_of_weights import save_proof_of_weights
from _validator.utils.uid import get_queryable_uids
from constants import (
    REQUEST_DELAY_SECONDS,
)
from execution_layer.circuit import Circuit, CircuitType
from utils import AutoUpdate, clean_temp_files, wandb_logger
from utils.gc_logging import log_responses as log_responses_gc


class ValidatorLoop:
    """
    Main loop for the validator node.

    The main loop for the validator. Handles everything from score updates to weight updates.
    """

    def __init__(self, config: ValidatorConfig):
        """
        Initialize the ValidatorLoop based on provided configuration.

        Args:
            config (bt.config): Bittensor configuration object.
        """
        self.config = config
        self.config.check_register()
        self.auto_update = AutoUpdate()
        self.score_manager = ScoreManager(
            self.config.metagraph, self.config.user_uid, self.config.full_path_score
        )
        self.response_processor = ResponseProcessor(
            metagraph=self.config.metagraph,
            score_manager=self.score_manager,
            user_uid=self.config.user_uid,
            rapidsnark_binary=self.config.rapidsnark_binary_path,
        )
        self.weights_manager = WeightsManager(
            self.config.subtensor,
            self.config.metagraph,
            self.config.wallet,
            self.config.user_uid,
        )
        self.last_pow_commit_block = 0
        self.api = ValidatorAPI(self.config)
        self.request_pipeline = RequestPipeline(
            self.config, self.score_manager, self.api
        )

        if self.config.bt_config.prometheus_monitoring:
            start_prometheus_logging(self.config.bt_config.prometheus_port)

    def run(self) -> NoReturn:
        """
        Run the main validator loop indefinitely.
        """
        bt.logging.debug("Validator started its running loop")

        while True:
            try:
                self._handle_auto_update()
                self.config.metagraph.sync(subtensor=self.config.subtensor)
                self.run_step()
            except KeyboardInterrupt:
                self._handle_keyboard_interrupt()
            except Exception as e:
                bt.logging.error(
                    f"Error in validator loop \n {e} \n {traceback.format_exc()}"
                )
            time.sleep(REQUEST_DELAY_SECONDS)

    def run_step(self) -> None:
        """
        Execute a single step of the validation process.
        """
        self.score_manager.sync_scores_uids(self.config.metagraph.uids.tolist())

        filtered_uids = list(get_queryable_uids(self.config.metagraph))

        random.shuffle(filtered_uids)

        requests: list[Request] = self.request_pipeline.prepare_requests(filtered_uids)

        if len(requests) == 0:
            bt.logging.error("No requests prepared")
            return

        bt.logging.info(
            f"\033[92m >> Sending {len(requests)} queries for proofs to miners in the subnet \033[0m"
        )

        try:
            start_time = time.time()
            responses: list[MinerResponse] = self._process_requests(requests)
            overhead_time: float = self._log_overhead_time(start_time)
            if not self.config.bt_config.disable_statistic_logging:
                log_responses_gc(
                    metagraph=self.config.metagraph,
                    hotkey=self.config.wallet.hotkey,
                    uid=self.config.user_uid,
                    responses=responses,
                    overhead_time=overhead_time,
                    block=self.config.subtensor.get_current_block(),
                    scores=self.score_manager.scores,
                )
        except RuntimeError as e:
            bt.logging.error(
                f"A runtime error occurred in the main validator loop\n{e}"
            )
            traceback.print_exc()
        except Exception as e:
            bt.logging.error(
                f"An error occurred in the main validator loop\n{e}\n{traceback.format_exc()}"
            )
        except KeyboardInterrupt:
            self._handle_keyboard_interrupt()

    def _handle_auto_update(self):
        """Handle automatic updates if enabled."""
        if not self.config.bt_config.no_auto_update:
            self.auto_update.try_update()
        else:
            bt.logging.debug("Automatic updates are disabled, skipping version check")

    def _process_requests(self, requests: list[Request]) -> list[MinerResponse]:
        """
        Process requests, update scores and weights.

        Args:
            requests (list): List of prepared requests.
        """
        loop = asyncio.get_event_loop()

        responses: list[Request] = loop.run_until_complete(
            query_axons(self.config.dendrite, requests)
        )

        processed_responses: list[MinerResponse] = (
            self.response_processor.process_responses(responses)
        )

        circuit: Circuit = requests[0].circuit

        if circuit.metadata.type == CircuitType.PROOF_OF_WEIGHTS:
            verified_responses = [
                r for r in processed_responses if r.verification_result
            ]
            if verified_responses:
                random_verified_response = random.choice(verified_responses)
                request_hash = requests[0].request_hash or hash_inputs(
                    requests[0].inputs
                )
                save_proof_of_weights(
                    public_signals=[random_verified_response.public_json],
                    proof=[random_verified_response.proof_content],
                    proof_filename=request_hash,
                )

                if requests[0].request_type == RequestType.RWR:
                    self.api.set_request_result(
                        request_hash,
                        {
                            "hash": request_hash,
                            "public_signals": random_verified_response.public_json,
                            "proof": random_verified_response.proof_content,
                        },
                    )

        self.score_manager.update_scores(processed_responses)
        self.weights_manager.update_weights(self.score_manager.scores)

        return processed_responses

    def _log_overhead_time(self, start_time) -> float:
        """
        Log the overhead time for processing.
        This is time that the validator spent verifying proofs, updating scores and performing other tasks.

        Args:
            start_time (float): Start time of processing.
        """
        end_time = time.time()
        overhead_time = end_time - start_time
        bt.logging.info(f"Overhead time: {overhead_time} seconds")
        wandb_logger.safe_log(
            {
                "overhead_time": overhead_time,
            }
        )
        log_validation_time(overhead_time)
        return overhead_time

    def _handle_keyboard_interrupt(self):
        """Handle keyboard interrupt by cleaning up and exiting."""
        bt.logging.success("Keyboard interrupt detected. Exiting validator.")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.api.stop())
        stop_prometheus_logging()
        clean_temp_files()
        sys.exit(0)
