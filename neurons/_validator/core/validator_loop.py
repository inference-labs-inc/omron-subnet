from __future__ import annotations

import asyncio
import random
import sys
import traceback
from typing import NoReturn

import bittensor as bt

from _validator.config import ValidatorConfig
from _validator.api import ValidatorAPI
from _validator.core.prometheus import (
    start_prometheus_logging,
    stop_prometheus_logging,
    log_request_metrics,
)
from _validator.core.request import Request
from _validator.core.request_pipeline import RequestPipeline
from _validator.core.response_processor import ResponseProcessor
from _validator.models.miner_response import MinerResponse
from _validator.scoring.score_manager import ScoreManager
from _validator.scoring.weights import WeightsManager
from _validator.utils.axon import query_single_axon
from _validator.models.request_type import RequestType
from _validator.utils.proof_of_weights import save_proof_of_weights
from _validator.utils.uid import get_queryable_uids
from constants import (
    REQUEST_DELAY_SECONDS,
    MAX_CONCURRENT_REQUESTS,
    ONE_MINUTE,
    FIVE_MINUTES,
    ONE_HOUR,
    BATCHED_PROOF_OF_WEIGHTS_MODEL_ID,
)
from execution_layer.circuit import CircuitType
from utils import AutoUpdate, clean_temp_files, with_rate_limit


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
            self.config.metagraph,
            self.score_manager,
            self.config.user_uid,
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

        self.request_queue = asyncio.Queue()
        self.active_requests: dict[int, asyncio.Task] = {}
        self.processed_uids: set[int] = set()
        self.queryable_uids: list[int] = []

        if self.config.bt_config.prometheus_monitoring:
            start_prometheus_logging(self.config.bt_config.prometheus_port)

    # Note that this rate limit is less than the weights rate limit
    # This is to reduce extra subtensor calls but ensure that we check
    # regularly with the updater
    @with_rate_limit(period=FIVE_MINUTES)
    def update_weights(self):
        self.weights_manager.update_weights(self.score_manager.scores)

    @with_rate_limit(period=ONE_HOUR)
    def sync_scores_uids(self):
        self.score_manager.sync_scores_uids(self.config.metagraph.uids.tolist())

    @with_rate_limit(period=ONE_HOUR)
    def sync_metagraph(self):
        self.config.metagraph.sync(subtensor=self.config.subtensor)

    @with_rate_limit(period=FIVE_MINUTES)
    def check_auto_update(self):
        self._handle_auto_update()

    @with_rate_limit(period=FIVE_MINUTES)
    def update_queryable_uids(self):
        self.queryable_uids = list(get_queryable_uids(self.config.metagraph))

    @with_rate_limit(period=ONE_MINUTE)
    def log_health(self):
        bt.logging.info(
            f"In-flight requests: {len(self.active_requests)} / {MAX_CONCURRENT_REQUESTS}"
        )
        bt.logging.debug(f"Processed UIDs: {len(self.processed_uids)}")
        bt.logging.debug(f"Queryable UIDs: {len(self.queryable_uids)}")

    def update_processed_uids(self):
        if len(self.processed_uids) >= len(self.queryable_uids):
            self.processed_uids.clear()

    async def update_active_requests(self):
        random.shuffle(self.queryable_uids)
        needed_requests = MAX_CONCURRENT_REQUESTS - len(self.active_requests)
        if needed_requests > 0:
            available_uids = [
                uid
                for uid in self.queryable_uids
                if uid not in self.processed_uids and uid not in self.active_requests
            ]

            if not available_uids:
                self.processed_uids.clear()
                return

            uid = available_uids[0]
            request = self.request_pipeline.prepare_single_request(uid)
            if request:
                task = asyncio.create_task(self._process_single_request(request))
                self.active_requests[uid] = task

        if self.active_requests:
            done, _ = await asyncio.wait(
                self.active_requests.values(),
                return_when=asyncio.FIRST_COMPLETED,
                timeout=0.1,
            )
            for task in done:
                uid, response = await task
                self.processed_uids.add(uid)
                del self.active_requests[uid]

                if response:
                    await self._handle_response(response)

        log_request_metrics(
            active_requests=len(self.active_requests),
            processed_uids=len(self.processed_uids),
        )

    async def run(self) -> NoReturn:
        """
        Run the main validator loop indefinitely.
        """
        bt.logging.success(
            f"Validator started on subnet {self.config.subnet_uid} using UID {self.config.user_uid}"
        )
        bt.logging.debug("Initializing request loop")

        while True:
            try:
                self.check_auto_update()
                self.sync_metagraph()
                self.sync_scores_uids()
                self.update_queryable_uids()
                self.update_processed_uids()
                self.log_health()
                await self.update_active_requests()
                await asyncio.sleep(0.1)

            except KeyboardInterrupt:
                self._handle_keyboard_interrupt()
            except Exception as e:
                bt.logging.error(
                    f"Error in validator loop \n {e} \n {traceback.format_exc()}"
                )
                await asyncio.sleep(REQUEST_DELAY_SECONDS)

    async def _process_single_request(
        self, request: Request
    ) -> tuple[int, MinerResponse | None]:
        """
        Process a single request and return the response.

        Args:
            request (Request): The request to process.

        Returns:
            tuple[int, MinerResponse | None]: The UID and processed response (if successful).
        """
        try:
            response = await query_single_axon(self.config.dendrite, request)
            if response:
                processed_response = self.response_processor.process_single_response(
                    response
                )
                return request.uid, processed_response
        except Exception as e:
            bt.logging.error(f"Error processing request for UID {request.uid}: {e}")

        return request.uid, None

    async def _handle_response(self, response: MinerResponse) -> None:
        """
        Handle a processed response, updating scores and weights as needed.

        Args:
            response (MinerResponse): The processed response to handle.
        """
        request_hash = response.input_hash
        if response.request_type == RequestType.RWR:
            if response.verification_result:
                self.api.set_request_result(
                    request_hash,
                    {
                        "hash": request_hash,
                        "public_signals": response.public_json,
                        "proof": response.proof_content,
                        "success": True,
                    },
                )
            else:
                self.api.set_request_result(
                    request_hash,
                    {
                        "success": False,
                    },
                )

        if response.verification_result:
            if (
                response.circuit.metadata.type == CircuitType.PROOF_OF_WEIGHTS
                and response.request_type == RequestType.RWR
            ) or response.circuit.id == BATCHED_PROOF_OF_WEIGHTS_MODEL_ID:
                save_proof_of_weights(
                    public_signals=[response.public_json],
                    proof=[response.proof_content],
                    proof_filename=request_hash,
                )

        self.score_manager.update_single_score(response, self.queryable_uids)

    def _handle_auto_update(self):
        """Handle automatic updates if enabled."""
        if not self.config.bt_config.no_auto_update:
            self.auto_update.try_update()
        else:
            bt.logging.debug("Automatic updates are disabled, skipping version check")

    def _handle_keyboard_interrupt(self):
        """Handle keyboard interrupt by cleaning up and exiting."""
        bt.logging.success("Keyboard interrupt detected. Exiting validator.")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.api.stop())
        stop_prometheus_logging()
        clean_temp_files()
        sys.exit(0)
