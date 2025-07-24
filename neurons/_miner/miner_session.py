import json
import os
import time
import traceback
import asyncio
from typing import Tuple, Union

import bittensor as bt
import websocket
from rich.console import Console
from rich.table import Table

import cli_parser
from _validator.models.request_type import RequestType
from constants import (
    STEAK,
    VALIDATOR_STAKE_THRESHOLD,
    CIRCUIT_TIMEOUT_SECONDS,
    NUM_MINER_GROUPS,
    MINER_RESET_WINDOW_BLOCKS,
)
from deployment_layer.circuit_store import circuit_store
from execution_layer.generic_input import GenericInput
from execution_layer.verified_model_session import VerifiedModelSession
from protocol import (
    QueryZkProof,
    ProofOfWeightsSynapse,
    Competition,
)
from utils import AutoUpdate, clean_temp_files, wandb_logger
from utils.rate_limiter import with_rate_limit
from utils.epoch import get_current_epoch_info, get_epoch_start_block
from .circuit_manager import CircuitManager
from .lightning_server import LightningServer
from utils.shuffle import get_shuffled_uids

COMPETITION_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "competition_circuit"
)


class MinerSession:

    axon: Union[bt.axon, None] = None
    lightning_server: Union[LightningServer, None] = None

    def __init__(self):
        self.configure()
        self.check_register(should_exit=True)
        self.auto_update = AutoUpdate()
        self.log_batch = []
        self.shuffled_uids = None
        self.last_shuffle_epoch = -1
        if cli_parser.config.disable_blacklist:
            bt.logging.warning(
                "Blacklist disabled, allowing all requests. Consider enabling to filter requests."
            )
        websocket.setdefaulttimeout(30)

    def start_axon(self):
        bt.logging.info(
            "Starting axon. Custom arguments include the following.\n"
            "Note that any null values will fallback to defaults, "
            f"which are usually sufficient. {cli_parser.config.axon}"
        )

        axon = bt.axon(wallet=self.wallet, config=cli_parser.config)
        bt.logging.info(f"Axon created: {axon.info()}")

        bt.logging.info("Attaching forward functions to axon...")
        axon.attach(forward_fn=self.queryZkProof, blacklist_fn=self.proof_blacklist)
        axon.attach(
            forward_fn=self.handle_pow_request,
            blacklist_fn=self.pow_blacklist,
        )
        axon.attach(
            forward_fn=self.handleCompetitionRequest,
            blacklist_fn=self.competition_blacklist,
        )

        bt.logging.info("Attached forward functions to axon")

        bt.logging.info(f"Starting axon server: {axon.info()}")
        axon.start()
        bt.logging.info(f"Started axon server: {axon.info()}")

        existing_axon = self.metagraph.axons[self.subnet_uid]

        if (
            existing_axon
            and existing_axon.port == axon.external_port
            and existing_axon.ip == axon.external_ip
        ):
            bt.logging.debug(
                f"Axon already serving on ip {axon.external_ip} and port {axon.external_port}"
            )
            return
        bt.logging.info(
            f"Serving axon on network: {self.subtensor.chain_endpoint} with netuid: {cli_parser.config.netuid}"
        )

        axon.serve(netuid=cli_parser.config.netuid, subtensor=self.subtensor)
        bt.logging.info(
            f"Served axon on network: {self.subtensor.chain_endpoint} with netuid: {cli_parser.config.netuid}"
        )

        self.axon = axon

    def start_lightning_server(self):
        """Initialize the Rust Lightning server"""
        lightning_port = getattr(cli_parser.config.axon, "port", 8091)
        external_ip = "0.0.0.0"
        bt.logging.info(
            f"🔧 Lightning server port calculation: axon.port={lightning_port}, using lightning_port={lightning_port}"
        )

        self.lightning_server = LightningServer(
            miner_session=self, host=external_ip, port=lightning_port
        )

        bt.logging.success(
            f"⚡ Lightning server initialized on {external_ip}:{lightning_port}"
        )
        bt.logging.info("🔥 Using Rust Lightning server for maximum performance!")
        bt.logging.info("⚡ Persistent connections enabled for validators")
        bt.logging.info("🤝 Optimized handshake protocol active")

    async def _start_lightning_server_async(self):
        """Start the Lightning server asynchronously"""
        await self.lightning_server.start()
        bt.logging.success("⚡ Lightning server started with persistent connections")
        bt.logging.info("🚀 Ready for high-throughput validator communication")

    async def _stop_lightning_server_async(self):
        """Stop the Lightning server asynchronously"""
        await self.lightning_server.stop()
        bt.logging.info("🔌 Lightning server stopped successfully")

    def run(self):
        bt.logging.info("Starting miner...")
        self.start_lightning_server()

        lightning_task = None
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            lightning_task = loop.create_task(self._run_lightning_server())
            bt.logging.success("⚡ Lightning server task started with Rust backend")

        except Exception as e:
            bt.logging.error(f"❌ Failed to start Lightning server task: {e}")
            lightning_task = None

        step = 0

        try:
            while True:
                step += 1
                try:

                    if step % 100 == 0:
                        if not cli_parser.config.no_auto_update:
                            self.auto_update.try_update()
                        else:
                            bt.logging.debug(
                                "Automatic updates are disabled, skipping version check"
                            )

                    if step % 20 == 0:
                        if len(self.log_batch) > 0:
                            bt.logging.debug(
                                f"Logging batch to WandB of size {len(self.log_batch)}"
                            )
                            for log in self.log_batch:
                                wandb_logger.safe_log(log)
                            self.log_batch = []
                        else:
                            bt.logging.debug("No logs to log to WandB")

                    if step % 600 == 0:
                        self.check_register()

                    if step % 24 == 0 and self.subnet_uid is not None:
                        table = Table(title=f"Miner Status (UID: {self.subnet_uid})")
                        table.add_column("Block", justify="center", style="cyan")
                        table.add_column("Stake", justify="center", style="cyan")
                        table.add_column("Rank", justify="center", style="cyan")
                        table.add_column("Trust", justify="center", style="cyan")
                        table.add_column("Consensus", justify="center", style="cyan")
                        table.add_column("Incentive", justify="center", style="cyan")
                        table.add_column("Emission", justify="center", style="cyan")
                        table.add_row(
                            str(self.metagraph.block.item()),
                            str(self.metagraph.S[self.subnet_uid]),
                            str(self.metagraph.R[self.subnet_uid]),
                            str(self.metagraph.T[self.subnet_uid]),
                            str(self.metagraph.C[self.subnet_uid]),
                            str(self.metagraph.I[self.subnet_uid]),
                            str(self.metagraph.E[self.subnet_uid]),
                        )
                        console = Console()
                        console.print(table)

                    if step % 500 == 0:
                        stats = self.lightning_server.get_connection_stats()
                        connections = stats.get("verified_connections", 0)
                        bt.logging.info(
                            f"⚡ Lightning server: {connections} active validator connections"
                        )

                    self.sync_metagraph()

                    time.sleep(1)

                except KeyboardInterrupt:
                    bt.logging.success("Miner killed via keyboard interrupt.")
                    break
                except Exception:
                    bt.logging.error(traceback.format_exc())
                    continue

        finally:
            bt.logging.info("Shutting down miner...")
            clean_temp_files()

            if lightning_task:
                try:
                    bt.logging.info("Stopping Lightning server...")
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(self._stop_lightning_server_async())
                    lightning_task.cancel()
                    bt.logging.success("Lightning server stopped")
                except Exception as e:
                    bt.logging.error(f"❌ Error stopping Lightning server: {e}")

    async def _run_lightning_server(self):
        """Run the Lightning server asynchronously"""
        try:
            bt.logging.info("🔧 Starting lightning server async initialization...")
            await self._start_lightning_server_async()
            bt.logging.info(
                "🚀 Lightning server async start completed, calling serve_forever..."
            )
            # Start the real QUIC server listening loop
            await self.lightning_server.serve_forever()
        except asyncio.CancelledError:
            bt.logging.debug("Lightning server task cancelled")
        except Exception as e:
            bt.logging.error(f"❌ Lightning server error: {e}")
            traceback.print_exc()

    def check_register(self, should_exit=False):
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            # flake8: noqa: E501
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {cli_parser.config.netuid}. Please register the hotkey using \n`btcli wallet register --wallet.name {self.wallet.name} --wallet.hotkey {self.wallet.hotkey_str} --subtensor.network {cli_parser.config.subtensor.network}`"
            )
            if should_exit:
                exit()
        else:
            self.subnet_uid = self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address
            )
            # flake8: noqa: E501
            bt.logging.info(
                f"Running miner on subnet: {cli_parser.config.netuid} with uid {self.subnet_uid} using network: {cli_parser.config.subtensor.network}"
            )

    def configure(self):
        cli_parser.init_config("miner")
        self.wallet = bt.wallet(config=cli_parser.config)
        self.subtensor = bt.subtensor(config=cli_parser.config)
        self.metagraph = self.subtensor.metagraph(cli_parser.config.netuid)
        bt.logging.info(f"Wallet: {self.wallet}")
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(f"Metagraph: {self.metagraph}")

    def sync_metagraph(self):
        try:
            self.metagraph.sync(subtensor=self.subtensor, lite=True)
        except Exception as e:
            bt.logging.error(f"Failed to sync metagraph: {e}")

    def perform_reset_check(self):
        bt.logging.debug("Performing reset check")
        try:
            current_epoch_info = get_current_epoch_info(cli_parser.config.netuid)
            current_epoch = current_epoch_info.epoch
            current_block = current_epoch_info.current_block

            if current_epoch != self.last_shuffle_epoch:
                bt.logging.info(
                    f"New epoch detected: {current_epoch}. Previous epoch: {self.last_shuffle_epoch}"
                )
                self.shuffled_uids = get_shuffled_uids(
                    cli_parser.config.netuid, current_epoch
                )
                self.last_shuffle_epoch = current_epoch

            miner_group = self.subnet_uid % NUM_MINER_GROUPS

            if miner_group in self.shuffled_uids:
                epoch_start_block = get_epoch_start_block(
                    cli_parser.config.netuid, current_epoch
                )
                blocks_from_start = current_block - epoch_start_block
                if blocks_from_start % MINER_RESET_WINDOW_BLOCKS == 0:
                    bt.logging.info(
                        f"Initiating reset for group {miner_group} "
                        f"at block {current_block} (epoch {current_epoch})"
                    )
                    self.perform_reset()
                else:
                    bt.logging.debug(
                        f"Miner group {miner_group} is not scheduled for reset at block {current_block}"
                    )
            else:
                bt.logging.debug(
                    f"Miner group {miner_group} is not in the shuffled UIDs for this epoch"
                )
        except Exception as e:
            bt.logging.error(f"Error performing reset check: {e}")

    @with_rate_limit(6.0)
    def proof_blacklist(self, synapse: QueryZkProof) -> Tuple[bool, str]:
        hotkey = synapse.dendrite.hotkey
        if hotkey not in self.metagraph.hotkeys:
            reason = f"Unrecognized hotkey {hotkey}"
            return True, reason

        uid = self.metagraph.hotkeys.index(hotkey)
        if not self.metagraph.validator_permit[uid]:
            if self.metagraph.S[uid] < VALIDATOR_STAKE_THRESHOLD * STEAK:
                reason = (
                    f"Hotkey {hotkey} has insufficient stake {self.metagraph.S[uid]}"
                )
                return True, reason

        return False, "Allowed!"

    @with_rate_limit(6.0)
    def pow_blacklist(self, synapse: ProofOfWeightsSynapse) -> Tuple[bool, str]:
        hotkey = synapse.dendrite.hotkey
        if hotkey not in self.metagraph.hotkeys:
            reason = f"Unrecognized hotkey {hotkey}"
            return True, reason

        uid = self.metagraph.hotkeys.index(hotkey)
        if not self.metagraph.validator_permit[uid]:
            if self.metagraph.S[uid] < VALIDATOR_STAKE_THRESHOLD * STEAK:
                reason = (
                    f"Hotkey {hotkey} has insufficient stake {self.metagraph.S[uid]}"
                )
                return True, reason

        return False, "Allowed!"

    @with_rate_limit(6.0)
    def competition_blacklist(self, synapse: Competition) -> Tuple[bool, str]:
        hotkey = synapse.dendrite.hotkey
        if hotkey not in self.metagraph.hotkeys:
            reason = f"Unrecognized hotkey {hotkey}"
            return True, reason

        uid = self.metagraph.hotkeys.index(hotkey)
        if not self.metagraph.validator_permit[uid]:
            if self.metagraph.S[uid] < VALIDATOR_STAKE_THRESHOLD * STEAK:
                reason = (
                    f"Hotkey {hotkey} has insufficient stake {self.metagraph.S[uid]}"
                )
                return True, reason

        return False, "Allowed!"

    def queryZkProof(self, synapse: QueryZkProof) -> QueryZkProof:
        """Process a zero-knowledge proof query request."""
        model_id = self._extract_model_id(synapse)

        try:
            if not self._is_circuit_store_available():
                self._set_error_response(
                    synapse, "Circuit store is empty or not initialized"
                )
                return self._log_and_return(synapse, model_id)

            if model_id is None:
                self._set_error_response(synapse, "No model_id found in query_input")
                return self._log_and_return(synapse, model_id)

            circuit = circuit_store.circuits.get(model_id)
            if not circuit:
                self._set_error_response(
                    synapse, f"Model {model_id} not found in circuit store"
                )
                return self._log_and_return(synapse, model_id)

            public_inputs = synapse.query_input.get("public_inputs")
            if not public_inputs:
                self._set_error_response(
                    synapse, f"No public_inputs found for model {model_id}"
                )
                return self._log_and_return(synapse, model_id)

            generic_input = GenericInput(RequestType.POC, public_inputs)
            verified_model_session = VerifiedModelSession(generic_input, circuit)

            result = verified_model_session.query(generic_input)

            if result.is_success:
                synapse.query_output = result.data
                bt.logging.info(f"Successfully processed proof for {model_id}")
            else:
                self._set_error_response(
                    synapse,
                    f"Failed to process proof for {model_id}: {result.error_message}",
                )

        except Exception as e:
            bt.logging.error(f"Error in queryZkProof for model {model_id}: {e}")
            self._set_error_response(
                synapse, f"Internal error processing model {model_id}"
            )

        return self._log_and_return(synapse, model_id)

    def _extract_model_id(self, synapse: QueryZkProof) -> str | None:
        """Extract model_id from synapse query_input."""
        if not synapse.query_input or "model_id" not in synapse.query_input:
            return None
        return synapse.query_input["model_id"]

    def _is_circuit_store_available(self) -> bool:
        """Check if circuit store is available and has circuits."""
        return hasattr(circuit_store, "circuits") and circuit_store.circuits

    def _set_error_response(self, synapse: QueryZkProof, error_message: str) -> None:
        """Set error response and log warning."""
        synapse.query_output = ""
        bt.logging.warning(error_message)

    def _log_and_return(
        self, synapse: QueryZkProof, model_id: str | None
    ) -> QueryZkProof:
        """Log the query result and return the synapse."""
        self.log_batch.append(
            {
                "model_id": model_id,
                "query_output": synapse.query_output,
                "timestamp": time.time(),
            }
        )
        return synapse

    def handle_pow_request(
        self, synapse: ProofOfWeightsSynapse
    ) -> ProofOfWeightsSynapse:
        """Process a proof-of-weights request."""
        verification_key_hash = synapse.verification_key_hash

        try:
            if not self._is_circuit_store_available():
                self._set_pow_error_response(
                    synapse, "Circuit store is empty or not initialized"
                )
                return synapse

            circuit = circuit_store.circuits.get(verification_key_hash)
            if not circuit:
                self._set_pow_error_response(
                    synapse,
                    f"Circuit {verification_key_hash} not found in circuit store",
                )
                return synapse

            try:
                input_data = json.loads(synapse.inputs)
            except json.JSONDecodeError as e:
                self._set_pow_error_response(
                    synapse, f"Invalid JSON in inputs for {verification_key_hash}: {e}"
                )
                return synapse

            generic_input = GenericInput(RequestType.POW, input_data)
            verified_model_session = VerifiedModelSession(generic_input, circuit)

            result = verified_model_session.query(generic_input)

            if result.is_success:
                synapse.proof = result.data.get("proof", "")
                synapse.public_signals = result.data.get("public_signals", "")
                bt.logging.info(
                    f"Successfully processed PoW for {verification_key_hash}"
                )
            else:
                self._set_pow_error_response(
                    synapse,
                    f"Failed to process PoW for {verification_key_hash}: {result.error_message}",
                )

        except Exception as e:
            bt.logging.error(
                f"Error in handle_pow_request for {verification_key_hash}: {e}"
            )
            self._set_pow_error_response(
                synapse, f"Internal error processing PoW for {verification_key_hash}"
            )

        return synapse

    def _set_pow_error_response(
        self, synapse: ProofOfWeightsSynapse, error_message: str
    ) -> None:
        """Set error response for PoW requests and log warning."""
        synapse.proof = ""
        synapse.public_signals = ""
        bt.logging.warning(error_message)

    def handleCompetitionRequest(self, synapse: Competition) -> Competition:
        try:
            circuit_manager = CircuitManager(
                wallet=self.wallet,
                dendrite=bt.dendrite(wallet=self.wallet),
                competition_id=synapse.id,
            )

            if synapse.hash:
                success = asyncio.run(
                    circuit_manager.download_files(
                        self.metagraph.axons[synapse.id], synapse.hash, COMPETITION_DIR
                    )
                )
                if success:
                    synapse.commitment = "Files downloaded successfully"
                else:
                    synapse.commitment = "Failed to download files"
            else:
                synapse.commitment = "No hash provided"

        except Exception as e:
            bt.logging.error(f"Error in handleCompetitionRequest: {e}")
            synapse.commitment = f"Error: {str(e)}"

        return synapse

    def perform_reset(self):
        bt.logging.info("🔄 Performing miner reset")

        try:
            bt.logging.info("🧹 Cleaning temporary files...")
            clean_temp_files()

            bt.logging.info("♻️ Clearing circuit store...")
            if hasattr(circuit_store, "circuits"):
                circuit_store.circuits.clear()

            if hasattr(circuit_store, "clear_cache"):
                circuit_store.clear_cache()

            bt.logging.info("🔄 Syncing metagraph after reset...")
            self.sync_metagraph()

            bt.logging.success("✅ Miner reset completed successfully")

        except Exception as e:
            bt.logging.error(f"❌ Error during miner reset: {e}")
            traceback.print_exc()
