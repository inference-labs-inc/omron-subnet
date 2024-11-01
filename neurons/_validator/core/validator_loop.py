from __future__ import annotations

import asyncio
import random
import sys
import time
import traceback
from typing import NoReturn

import bittensor as bt


from _validator.config import ValidatorConfig
from _validator.core.response_processor import ResponseProcessor
from _validator.scoring.score_manager import ScoreManager
from _validator.scoring.weights import WeightsManager
from _validator.utils.axon import query_axons
from _validator.utils.proof_of_weights import (
    log_and_commit_proof,
    save_proof_of_weights,
)
from _validator.utils.uid import get_queryable_uids
from _validator.core.api import ValidatorAPI
from constants import (
    ONCHAIN_PROOF_OF_WEIGHTS_ENABLED,
    PROOF_OF_WEIGHTS_INTERVAL,
    REQUEST_DELAY_SECONDS,
    SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
    SINGLE_PROOF_OF_WEIGHTS_MODEL_ID_JOLT,
)
from _validator.utils.api import hash_inputs
from utils import AutoUpdate, clean_temp_files, wandb_logger
from _validator.core.request_pipeline import RequestPipeline


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
        self.score_manager = ScoreManager(self.config.metagraph, self.config.user_uid)
        self.response_processor = ResponseProcessor(
            self.config.metagraph, self.score_manager, self.config.user_uid
        )
        self.log_pow = False
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

        requests = self.request_pipeline.prepare_requests(filtered_uids)

        if len(requests) == 0:
            bt.logging.error("No requests prepared")
            return

        bt.logging.info(
            f"\033[92m >> Sending {len(requests)} queries for proofs to miners in the subnet \033[0m"
        )

        try:
            start_time = time.time()
            self._process_requests(requests)
            self._log_overhead_time(start_time)
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

    def _process_requests(self, requests):
        """
        Process requests, update scores and weights.

        Args:
            requests (list): List of prepared requests.
        """
        loop = asyncio.get_event_loop()

        responses = loop.run_until_complete(query_axons(self.config.dendrite, requests))

        processed_responses = self.response_processor.process_responses(responses)
        if requests[0].get("model_id") not in [
            SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
            SINGLE_PROOF_OF_WEIGHTS_MODEL_ID_JOLT,
        ]:
            verified_responses = [
                r for r in processed_responses if r.verification_result
            ]
            if verified_responses:
                random_verified_response = random.choice(verified_responses)
                save_proof_of_weights(
                    public_signals=[random_verified_response.public_json],
                    proof=[random_verified_response.proof_content],
                    proof_filename=hash_inputs(requests[0].get("inputs")),
                )

        self.score_manager.update_scores(processed_responses)

        if self.log_pow and self.config.bt_config.get(
            "enable_pow", ONCHAIN_PROOF_OF_WEIGHTS_ENABLED
        ):
            log_and_commit_proof(
                self.config.wallet.hotkey,
                self.config.subtensor,
                self.response_processor.completed_proof_of_weights_queue,
            )
            self.last_pow_commit_block = self.config.subtensor.get_current_block()
            self.response_processor.completed_proof_of_weights_queue = []
            self.log_pow = False

        if self.weights_manager.update_weights(self.score_manager.scores):
            if (
                self.config.bt_config.get(
                    "enable_pow", ONCHAIN_PROOF_OF_WEIGHTS_ENABLED
                )
                and self.last_pow_commit_block
                + int(
                    self.config.bt_config.get(
                        "pow_target_interval", PROOF_OF_WEIGHTS_INTERVAL
                    )
                )
                < self.config.subtensor.get_current_block()
                and len(self.response_processor.completed_proof_of_weights_queue)
                and not self.log_pow
            ):
                # Log PoW during the next iteration
                self.log_pow = True

    def _log_overhead_time(self, start_time):
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

    def _handle_keyboard_interrupt(self):
        """Handle keyboard interrupt by cleaning up and exiting."""
        bt.logging.success("Keyboard interrupt detected. Exiting validator.")
        self.api.stop()
        clean_temp_files()
        sys.exit(0)
