from __future__ import annotations

import asyncio
import sys
import traceback
import time
from typing import NoReturn
import concurrent.futures

import bittensor as bt

from _validator.config import ValidatorConfig
from _validator.api import ValidatorAPI
from _validator.core.prometheus import (
    start_prometheus_logging,
    stop_prometheus_logging,
    log_system_metrics,
    log_queue_metrics,
    log_weight_update,
    log_score_change,
    log_error,
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
from utils import AutoUpdate, clean_temp_files, with_rate_limit
from utils.gc_logging import log_responses as gc_log_responses
from _validator.competitions.competition import Competition
from _validator.utils.logging import log_responses as console_log_responses
from constants import (
    LOOP_DELAY_SECONDS,
    EXCEPTION_DELAY_SECONDS,
    MAX_CONCURRENT_REQUESTS,
    ONE_MINUTE,
    FIVE_MINUTES,
    ONE_HOUR,
)


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

        try:
            competition_id = 1
            self.competition = Competition(
                competition_id,
                self.config.metagraph,
                self.config.subtensor,
            )
        except Exception as e:
            bt.logging.warning(
                f"Failed to initialize competition, continuing without competition support: {e}"
            )
            self.competition = None

        self.score_manager = ScoreManager(
            self.config.metagraph,
            self.config.user_uid,
            self.config.full_path_score,
            self.competition,
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
        self.last_competition_sync = 0
        self.is_syncing_competition = False

        self.request_queue = asyncio.Queue()
        self.active_tasks: dict[int, asyncio.Task] = {}
        self.processed_uids: set[int] = set()
        self.queryable_uids: list[int] = []
        self.last_response_time = time.time()

        self._should_run = True

        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)
        self.response_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=16
        )

        self.recent_responses: list[MinerResponse] = []

        if self.config.bt_config.prometheus_monitoring:
            start_prometheus_logging(self.config.bt_config.prometheus_port)

    @with_rate_limit(period=FIVE_MINUTES)
    def update_weights(self):
        start_time = time.time()
        try:
            self.weights_manager.update_weights(self.score_manager.scores)
            duration = time.time() - start_time
            log_weight_update(duration, success=True)
        except Exception as e:
            log_weight_update(0.0, success=False, failure_reason=str(e))
            log_error("weight_update", "weights_manager", str(e))
            raise

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

    @with_rate_limit(period=ONE_MINUTE / 4)
    def log_health(self):
        bt.logging.info(
            f"In-flight requests: {len(self.active_tasks)} / {MAX_CONCURRENT_REQUESTS}"
        )
        bt.logging.debug(f"Processed UIDs: {len(self.processed_uids)}")
        bt.logging.debug(f"Queryable UIDs: {len(self.queryable_uids)}")

        log_system_metrics()
        queue_size = self.request_queue.qsize()
        est_latency = (
            queue_size * (LOOP_DELAY_SECONDS / MAX_CONCURRENT_REQUESTS)
            if queue_size > 0
            else 0
        )
        log_queue_metrics(queue_size, est_latency)

    def update_processed_uids(self):
        if len(self.processed_uids) >= len(self.queryable_uids):
            self.processed_uids.clear()

    @with_rate_limit(period=ONE_MINUTE)
    async def log_responses(self):
        if self.recent_responses:
            console_log_responses(self.recent_responses)

            try:
                block = (
                    self.config.metagraph.block.item()
                    if self.config.metagraph.block is not None
                    else 0
                )
                _ = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    lambda: gc_log_responses(
                        self.config.metagraph,
                        self.config.wallet.hotkey,
                        self.config.user_uid,
                        self.recent_responses,
                        (
                            time.time() - self.last_response_time
                            if hasattr(self, "last_response_time")
                            else 0
                        ),
                        block,
                        self.score_manager.scores,
                    ),
                )
            except Exception as e:
                bt.logging.error(f"Error in GC logging: {e}")

            self.last_response_time = time.time()
            self.recent_responses = []

    @with_rate_limit(period=ONE_HOUR)
    async def sync_competition(self):
        if not self.competition:
            return

        if self.is_syncing_competition:
            return

        try:
            self.is_syncing_competition = True
            bt.logging.info("Starting competition sync and evaluation...")

            for task in self.active_tasks.values():
                task.cancel()
            self.active_tasks.clear()

            while not self.request_queue.empty():
                try:
                    self.request_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            self.competition.sync_and_eval()

            bt.logging.success("Competition sync and evaluation complete")

        except Exception as e:
            bt.logging.error(f"Error in competition sync: {e}")
            traceback.print_exc()
        finally:
            self.is_syncing_competition = False

    async def maintain_request_pool(self):
        while True:
            try:
                if not self.is_syncing_competition:
                    slots_available = MAX_CONCURRENT_REQUESTS - len(self.active_tasks)
                    if slots_available > 0:
                        available_uids = [
                            uid
                            for uid in self.queryable_uids
                            if uid not in self.processed_uids
                            and uid not in self.active_tasks
                        ]

                        for uid in available_uids[:slots_available]:
                            request = self.request_pipeline.prepare_single_request(uid)
                            if request:
                                task = asyncio.create_task(
                                    self._process_single_request(request)
                                )
                                self.active_tasks[uid] = task
                                task.add_done_callback(
                                    lambda t, uid=uid: self._handle_completed_task(
                                        t, uid
                                    )
                                )

                await asyncio.sleep(0)
            except Exception as e:
                bt.logging.error(f"Error maintaining request pool: {e}")
                traceback.print_exc()
                await asyncio.sleep(EXCEPTION_DELAY_SECONDS)

    def _handle_completed_task(self, task: asyncio.Task, uid: int):
        try:
            self.processed_uids.add(uid)
        except Exception as e:
            bt.logging.error(f"Error in task for UID {uid}: {e}")
            traceback.print_exc()
        finally:
            if uid in self.active_tasks:
                del self.active_tasks[uid]

    async def run_periodic_tasks(self):
        while True:
            try:
                if not self.is_syncing_competition:
                    self.check_auto_update()
                    self.sync_metagraph()
                    self.sync_scores_uids()
                    self.update_weights()
                    self.update_queryable_uids()
                    self.update_processed_uids()
                    self.log_health()
                    await self.log_responses()
                await self.sync_competition()
                await asyncio.sleep(LOOP_DELAY_SECONDS)
            except Exception as e:
                bt.logging.error(f"Error in periodic tasks: {e}")
                traceback.print_exc()
                await asyncio.sleep(EXCEPTION_DELAY_SECONDS)

    async def run(self) -> NoReturn:
        """
        Run the main validator loop indefinitely.
        """
        bt.logging.success(
            f"Validator started on subnet {self.config.subnet_uid} using UID {self.config.user_uid}"
        )

        try:
            await asyncio.gather(
                self.maintain_request_pool(),
                self.run_periodic_tasks(),
            )
        except KeyboardInterrupt:
            self._should_run = False
            self._handle_keyboard_interrupt()
        except Exception as e:
            bt.logging.error(f"Fatal error in validator loop: {e}")
            raise

    async def _process_single_request(self, request: Request) -> Request:
        """
        Process a single request and return the response.
        """
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: query_single_axon(self.config.dendrite, request),
            )
            response = await response
            processed_response = await asyncio.get_event_loop().run_in_executor(
                self.response_thread_pool,
                self.response_processor.process_single_response,
                response,
            )
            if processed_response:
                await self._handle_response(processed_response)
        except Exception as e:
            bt.logging.error(f"Error processing request for UID {request.uid}: {e}")
            traceback.print_exc()
            log_error("request_processing", "axon_query", str(e))
        return request

    async def _handle_response(self, response: MinerResponse) -> None:
        """
        Handle a processed response, updating scores and weights as needed.

        Args:
            response (MinerResponse): The processed response to handle.
        """
        try:
            request_hash = response.input_hash
            self.recent_responses.append(response)
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

            if response.verification_result and response.save:
                save_proof_of_weights(
                    public_signals=[response.public_json],
                    proof=[response.proof_content],
                    proof_filename=request_hash,
                )

            old_score = (
                float(self.score_manager.scores[response.uid])
                if response.uid < len(self.score_manager.scores)
                else 0.0
            )
            self.score_manager.update_single_score(response, self.queryable_uids)
            new_score = (
                float(self.score_manager.scores[response.uid])
                if response.uid < len(self.score_manager.scores)
                else 0.0
            )
            log_score_change(old_score, new_score)

        except Exception as e:
            bt.logging.error(f"Error handling response: {e}")
            traceback.print_exc()
            log_error("response_handling", "response_processor", str(e))

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
