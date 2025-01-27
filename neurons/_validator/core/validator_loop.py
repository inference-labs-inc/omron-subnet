from __future__ import annotations

import asyncio
import sys
import traceback
from typing import NoReturn
import time
from dataclasses import dataclass

import bittensor as bt
import psutil

from _validator.config import ValidatorConfig
from _validator.api import ValidatorAPI
from _validator.core.prometheus import (
    start_prometheus_logging,
    stop_prometheus_logging,
    log_request_metrics,
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
    MAX_CONCURRENT_REQUESTS,
    FIVE_MINUTES,
    ONE_HOUR,
)
from execution_layer.circuit import CircuitType
from utils import AutoUpdate, clean_temp_files


@dataclass
class ResponseEvent:
    """Event wrapper for miner responses"""

    response: MinerResponse


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

        self.event_queue = asyncio.Queue()
        self.running = True
        self.processed_tasks: set[asyncio.Task] = set()
        self.processed_uids: set[int] = set()
        self.queryable_uids: list[int] = []

        self.completed_requests = 0
        self.last_completed_check = time.time()
        self.total_responses = 0
        self.total_processed = 0
        self.start_time = time.time()
        self.total_requests = 0

        if self.config.bt_config.prometheus_monitoring:
            start_prometheus_logging(self.config.bt_config.prometheus_port)

    async def update_weights(self):
        """Updates network weights based on current scores"""
        self.weights_manager.update_weights(self.score_manager.scores)

    async def sync_scores_uids(self):
        """Syncs scores with current network UIDs"""
        self.score_manager.sync_scores_uids(self.config.metagraph.uids.tolist())

    async def sync_metagraph(self):
        """Syncs local metagraph with network state"""
        self.config.metagraph.sync(subtensor=self.config.subtensor)

    async def check_auto_update(self):
        """Checks and performs auto-updates if enabled"""
        self._handle_auto_update()

    async def update_queryable_uids(self):
        """Updates list of UIDs that can be queried"""
        self.queryable_uids = list(get_queryable_uids(self.config.metagraph))

    async def log_health(self):
        """Logs validator health metrics"""
        current_time = time.time()
        total_runtime_minutes = (current_time - self.start_time) / 60
        avg_responses_per_min = (
            int(self.total_processed / total_runtime_minutes)
            if total_runtime_minutes > 0
            else 0
        )
        uptime_hours = (current_time - self.start_time) / 3600

        bt.logging.info(
            f"In-flight: {len(self.processed_tasks)}/{MAX_CONCURRENT_REQUESTS} | "
            f"RPM: {avg_responses_per_min} | "
            f"Processed UIDs: {len(self.processed_uids)} | "
            f"Lifetime - Total: {self.total_processed} Success: {self.total_responses} ({uptime_hours:.1f}h)"
        )
        bt.logging.debug(f"Queryable UIDs: {len(self.queryable_uids)}")

    def update_processed_uids(self):
        if len(self.processed_uids) >= len(self.queryable_uids):
            self.processed_uids.clear()

    async def update_active_requests(self):
        """Update active requests to maintain MAX_CONCURRENT_REQUESTS."""
        done_tasks = {task for task in self.processed_tasks if task.done()}
        for task in done_tasks:
            try:
                uid, response = await task
                self.processed_uids.add(uid)
                self.total_processed += 1
                self.total_requests += 1
                bt.logging.debug(f"Got response from UID {uid}")
                if response:
                    await self.event_queue.put(ResponseEvent(response))
            except Exception as e:
                bt.logging.error(f"Task failed with error: {str(e)}")
            finally:
                self.processed_tasks.remove(task)

        if len(self.processed_tasks) >= MAX_CONCURRENT_REQUESTS:
            return

        while len(self.processed_tasks) < MAX_CONCURRENT_REQUESTS:
            available_uids = [
                uid
                for uid in self.queryable_uids
                if uid not in self.processed_uids
                and not any(uid == int(t.get_name()) for t in self.processed_tasks)
            ]

            if not available_uids:
                if len(self.processed_uids) > 0:
                    bt.logging.debug(
                        "Resetting processed UIDs to maintain request volume"
                    )
                    self.processed_uids.clear()
                    continue
                break

            uid = available_uids[0]
            if request := self.request_pipeline.prepare_single_request(uid):
                task = asyncio.create_task(
                    self._process_single_request(request), name=str(uid)
                )
                self.processed_tasks.add(task)
                bt.logging.debug(f"Added request for UID {uid}")
            else:
                continue

        if len(self.processed_tasks) < MAX_CONCURRENT_REQUESTS:
            bt.logging.warning(
                f"Running below capacity: {len(self.processed_tasks)}/{MAX_CONCURRENT_REQUESTS} requests"
            )

        process = psutil.Process()
        log_request_metrics(
            active_requests=len(self.processed_tasks),
            processed_uids=len(self.processed_uids),
            memory_bytes=process.memory_info().rss,
        )

    async def run(self) -> NoReturn:
        """Run the main validator loop using event-driven architecture."""
        bt.logging.success(
            f"Validator started on subnet {self.config.subnet_uid} using UID {self.config.user_uid}"
        )

        tasks = [
            self._schedule_periodic(self.sync_metagraph, ONE_HOUR),
            self._schedule_periodic(self.sync_scores_uids, ONE_HOUR),
            self._schedule_periodic(self.update_weights, FIVE_MINUTES),
            self._schedule_periodic(self.update_queryable_uids, FIVE_MINUTES),
            self._schedule_periodic(self.check_auto_update, FIVE_MINUTES),
            self._schedule_periodic(self.log_health, 1.0),
            self._process_requests(),
            self._process_events(),
        ]

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self._handle_keyboard_interrupt()
        except Exception as e:
            bt.logging.error(
                f"Error in validator loop \n {e} \n {traceback.format_exc()}"
            )
            raise

    async def _schedule_periodic(self, coro, interval: float):
        """Schedule a coroutine to run periodically."""
        while self.running:
            try:
                await coro()
            except Exception as e:
                bt.logging.error(f"Error in periodic task {coro.__name__}: {e}")
            await asyncio.sleep(interval)

    async def _process_requests(self):
        """Continuously process requests."""
        while self.running:
            try:
                await self.update_active_requests()
                await asyncio.sleep(0.01)
            except Exception as e:
                bt.logging.error(f"Error processing requests: {e}")

    async def _process_events(self):
        """Process events from the event queue."""
        while self.running:
            try:
                event = await self.event_queue.get()
                await self._handle_event(event)
                self.event_queue.task_done()
            except Exception as e:
                bt.logging.error(f"Error processing event: {e}")

    async def _handle_event(self, event):
        """Handle different types of events."""
        if isinstance(event, ResponseEvent):
            await self._handle_response(event.response)

    async def _process_single_request(
        self, request: Request
    ) -> tuple[int, MinerResponse | None]:
        try:
            response = await query_single_axon(self.config.dendrite, request)
            if response:
                return request.uid, self.response_processor.process_single_response(
                    response
                )
        except Exception as e:
            bt.logging.error(f"Unexpected error processing UID {request.uid}: {str(e)}")
            bt.logging.debug(traceback.format_exc())
            log_network_error("unknown_error")

        return request.uid, None

    async def _handle_response(self, response: MinerResponse) -> None:
        """Handle a processed response, updating scores and weights."""
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

        if response.verification_result:
            self.total_responses += 1

    def _handle_auto_update(self):
        if not self.config.bt_config.no_auto_update:
            self.auto_update.try_update()
        else:
            bt.logging.debug("Automatic updates are disabled, skipping version check")

    def stop(self):
        """Gracefully stop the validator loop."""
        self.running = False
        if self.config.bt_config.prometheus_monitoring:
            stop_prometheus_logging()
        clean_temp_files()

    def _handle_keyboard_interrupt(self):
        """Handle keyboard interrupt by cleaning up and exiting."""
        bt.logging.success("Keyboard interrupt detected. Exiting validator.")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.api.stop())
        self.stop()
        sys.exit(0)
