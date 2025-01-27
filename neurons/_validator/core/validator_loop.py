from __future__ import annotations

import asyncio
import sys
import traceback
from typing import NoReturn

import bittensor as bt
import psutil

from _validator.config import ValidatorConfig
from _validator.api import ValidatorAPI
from _validator.core.prometheus import (
    start_prometheus_logging,
    stop_prometheus_logging,
    log_request_metrics,
    log_timeout,
    log_network_error,
    log_verification_ratio,
    log_proof_sizes,
    log_verification_failure,
    log_response_times,
    log_validation_time,
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
    VALIDATOR_REQUEST_TIMEOUT_SECONDS,
)
from execution_layer.circuit import CircuitType
from utils import AutoUpdate, clean_temp_files, with_rate_limit


class ValidatorLoop:
    """
    Core validator node orchestrator responsible for:
    - Managing concurrent miner requests
    - Updating network weights and scores
    - Processing responses and proofs
    - Handling auto-updates and prometheus metrics

    The loop maintains a pool of active requests up to MAX_CONCURRENT_REQUESTS,
    processes responses asynchronously, and updates network state based on
    configurable intervals.
    """

    def __init__(self, config: ValidatorConfig):
        """
        Initialize the ValidatorLoop based on provided configuration.

        Args:
            config (ValidatorConfig): Validator configuration object.
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

        # Request management
        self.active_requests: dict[int, asyncio.Task] = {}
        self.processed_uids: set[int] = set()
        self.queryable_uids: list[int] = []

        if self.config.bt_config.prometheus_monitoring:
            start_prometheus_logging(self.config.bt_config.prometheus_port)

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
        # Process completed requests first
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

        needed_requests = MAX_CONCURRENT_REQUESTS - len(self.active_requests)
        if needed_requests <= 0:
            return

        available_uids = [
            uid
            for uid in self.queryable_uids
            if uid not in self.processed_uids and uid not in self.active_requests
        ][:needed_requests]

        for uid in available_uids:
            if request := self.request_pipeline.prepare_single_request(uid):
                task = asyncio.create_task(self._process_single_request(request))
                self.active_requests[uid] = task

        # Log request metrics with memory usage
        process = psutil.Process()
        log_request_metrics(
            active_requests=len(self.active_requests),
            processed_uids=len(self.processed_uids),
            memory_bytes=process.memory_info().rss,
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
            response = await asyncio.wait_for(
                query_single_axon(self.config.dendrite, request),
                timeout=VALIDATOR_REQUEST_TIMEOUT_SECONDS,
            )
            if response:
                processed_response = self.response_processor.process_single_response(
                    response
                )
                return request.uid, processed_response

        except asyncio.TimeoutError:
            bt.logging.warning(f"Request to UID {request.uid} timed out")
            log_timeout(request.circuit.metadata.name if request.circuit else "unknown")
        except ConnectionError as e:
            bt.logging.warning(f"Network error for UID {request.uid}: {str(e)}")
            log_network_error("connection_error")
        except Exception as e:
            bt.logging.error(f"Unexpected error processing UID {request.uid}: {str(e)}")
            bt.logging.debug(traceback.format_exc())
            log_network_error("unknown_error")

        return request.uid, None

    async def _handle_response(self, response: MinerResponse) -> None:
        """
        Handle a processed response, updating scores and weights as needed.

        Args:
            response (MinerResponse): The processed response to handle.
        """
        model_name = response.circuit.metadata.name if response.circuit else "unknown"

        if response.verification_result and response.proof_content:
            if response.circuit.metadata.type == CircuitType.PROOF_OF_WEIGHTS:
                request_hash = response.input_hash
                save_proof_of_weights(
                    public_signals=[response.public_json],
                    proof=[response.proof_content],
                    proof_filename=request_hash,
                )

                if response.request_type == RequestType.RWR:
                    self.api.set_request_result(
                        request_hash,
                        {
                            "hash": request_hash,
                            "public_signals": response.public_json,
                            "proof": response.proof_content,
                        },
                    )

            if response.proof_content:
                log_proof_sizes([len(response.proof_content)], model_name)
        else:
            if response.error:
                log_verification_failure(model_name, response.error)

        if response.response_time:
            log_response_times([response.response_time], model_name)
        if response.verification_time:
            log_validation_time(response.verification_time)

        if response.circuit:
            log_verification_ratio(
                response.circuit.evaluation_data.verification_ratio, model_name
            )

        self.score_manager.update_single_score(
            response, queryable_uids=set(self.queryable_uids)
        )

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
