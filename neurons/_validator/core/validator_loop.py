from __future__ import annotations

import asyncio
import random
import copy
import sys
import time
import traceback
from typing import NoReturn

import bittensor as bt

from _validator.config.config import ValidatorConfig
from _validator.core.response_processor import ResponseProcessor
from _validator.scoring.score_manager import ScoreManager
from _validator.utils.proof_of_weights import log_and_commit_proof
from _validator.utils.uid import get_queryable_uids
from _validator.pow.proof_of_weights_handler import ProofOfWeightsHandler
from _validator.utils.axon import query_axons
from _validator.scoring.weights import WeightsManager
from utils import AutoUpdate, clean_temp_files, wandb_logger
from constants import (
    BATCHED_PROOF_OF_WEIGHTS_MODEL_ID,
    ONCHAIN_PROOF_OF_WEIGHTS_ENABLED,
    PROOF_OF_WEIGHTS_INTERVAL,
    REQUEST_DELAY_SECONDS,
)


class ValidatorLoop:
    """
    Main loop for the validator node.

    The main loop for the validator. Handles everything from score updates to weight updates.
    """

    def __init__(self, config: bt.config):
        """
        Initialize the ValidatorLoop based on provided configuration.

        Args:
            config (bt.config): Bittensor configuration object.
        """
        self.config = ValidatorConfig(config)
        self.config.check_register()
        self.auto_update = AutoUpdate()
        self.score_manager = ScoreManager(self.config.metagraph, self.config.user_uid)
        self.response_processor = ResponseProcessor(
            self.config.metagraph, self.score_manager, self.config.user_uid
        )
        self.weights_manager = WeightsManager(
            self.config.subtensor,
            self.config.metagraph,
            self.config.wallet,
            self.config.user_uid,
        )
        self.last_pow_commit_block = 0

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

    def prepare_requests(self, filtered_uids):
        """
        Prepare requests for querying miners, ensuring all requests are identical
        except for the last UID in the validator_uid list.

        Args:
            filtered_uids (list): List of filtered UIDs to query.

        Returns:
            list: List of prepared requests.
        """

        base_request = ProofOfWeightsHandler.prepare_pow_request(
            0, self.score_manager.proof_of_weights_queue, self.config.subnet_uid
        )

        requests = []
        for uid in filtered_uids:
            axon = self.config.metagraph.axons[uid]
            pow_request = copy.deepcopy(base_request)

            # Update the last UID in the validator_uid list with the miner's UID
            if "inputs" in pow_request and "validator_uid" in pow_request["inputs"]:
                pow_request["inputs"]["validator_uid"][-1] = uid
            else:
                bt.logging.warning(
                    f"Unable to update validator_uid for miner {uid}. Check the structure of pow_request."
                )

            bt.logging.trace(
                f"Prepared request for miner {uid} with axon {axon}: {pow_request}"
            )
            requests.append({"uid": uid, "axon": axon, **pow_request})

        if (
            requests
            and requests[0].get("model_id") == BATCHED_PROOF_OF_WEIGHTS_MODEL_ID
        ):
            self.score_manager.clear_proof_of_weights_queue()

        return requests

    def run_step(self) -> None:
        """
        Execute a single step of the validation process.
        """
        uids = self.config.metagraph.uids.tolist()
        self.score_manager.sync_scores_uids(uids)

        filtered_uids = list(get_queryable_uids(self.config.metagraph, uids))

        random.shuffle(filtered_uids)

        requests = self.prepare_requests(filtered_uids)

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
        if not self.config.config.no_auto_update:
            self.auto_update.try_update()
        else:
            bt.logging.info("Automatic updates are disabled, skipping version check")

    def _process_requests(self, requests):
        """
        Process requests, update scores and weights.

        Args:
            requests (list): List of prepared requests.
        """
        loop = asyncio.get_event_loop()

        responses = loop.run_until_complete(query_axons(self.config.dendrite, requests))

        processed_responses = self.response_processor.process_responses(responses)

        self.score_manager.update_scores(processed_responses)

        if self.weights_manager.update_weights(self.score_manager.scores):
            if (
                self.config.config.get("enable_pow", ONCHAIN_PROOF_OF_WEIGHTS_ENABLED)
                and self.last_pow_commit_block
                + int(
                    self.config.config.get(
                        "pow_target_interval", PROOF_OF_WEIGHTS_INTERVAL
                    )
                )
                < self.config.subtensor.get_current_block()
            ):
                log_and_commit_proof(
                    self.config.wallet.hotkey,
                    self.config.subtensor,
                    [self.response_processor.completed_proof_of_weights_queue[-1]],
                )
                self.last_pow_commit_block = self.config.subtensor.get_current_block()

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
        clean_temp_files()
        sys.exit(0)
