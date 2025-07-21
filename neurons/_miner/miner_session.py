from __future__ import annotations
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
from .quic_server import LightningServer
from utils.shuffle import get_shuffled_uids

COMPETITION_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "competition_circuit"
)


class MinerSession:

    axon: Union[bt.axon, None] = None
    quic_server: Union[LightningServer, None] = None

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

    def start_quic_server(self):
        try:
            quic_port = getattr(cli_parser.config.axon, "port", 8091) + 1
            external_ip = (
                getattr(self.axon, "external_ip", "0.0.0.0") if self.axon else "0.0.0.0"
            )

            self.quic_server = LightningServer(
                miner_session=self, host=external_ip, port=quic_port
            )

            bt.logging.info(
                f"Lightning server will be available at {external_ip}:{quic_port}"
            )
            bt.logging.info("Lightning server provides Lightning client compatibility")

        except Exception as e:
            bt.logging.error(f"Failed to initialize Lightning server: {e}")
            traceback.print_exc()
            self.quic_server = None

    async def _start_quic_server_async(self):
        if self.quic_server:
            try:
                await self.quic_server.start()
            except Exception as e:
                bt.logging.error(f"Failed to start Lightning server: {e}")
                traceback.print_exc()

    async def _stop_quic_server_async(self):
        if self.quic_server:
            try:
                await self.quic_server.stop()
            except Exception as e:
                bt.logging.error(f"Error stopping Lightning server: {e}")

    def perform_reset(self):
        bt.logging.info("Performing coordinated reset")
        try:
            commitment_info = [{"ResetBondsFlag": b""}]
            call = self.subtensor.substrate.compose_call(
                call_module="Commitments",
                call_function="set_commitment",
                call_params={
                    "netuid": cli_parser.config.netuid,
                    "info": {"fields": [commitment_info]},
                },
            )
            success, message = self.subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=self.wallet,
                sign_with="hotkey",
                period=MINER_RESET_WINDOW_BLOCKS,
            )
            if not success:
                bt.logging.error(f"Failed to perform reset: {message}")
            else:
                bt.logging.success("Successfully performed reset")
        except Exception as e:
            bt.logging.error(f"Error performing reset: {e}")

    def run(self):
        bt.logging.info("Starting miner...")
        self.start_axon()
        self.start_quic_server()

        quic_task = None
        if self.quic_server:
            try:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                quic_task = loop.create_task(self._run_quic_server())
                bt.logging.success("Lightning server task started")
            except Exception as e:
                bt.logging.error(f"Failed to start Lightning server task: {e}")
                quic_task = None

        step = 0

        try:
            while True:
                step += 1
                try:
                    if step % 10 == 0:
                        self.perform_reset_check()

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

            if quic_task and self.quic_server:
                try:
                    bt.logging.info("Stopping Lightning server...")
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(self._stop_quic_server_async())
                    quic_task.cancel()
                    bt.logging.success("Lightning server stopped")
                except Exception as e:
                    bt.logging.error(f"Error stopping Lightning server: {e}")

    async def _run_quic_server(self):
        try:
            await self._start_quic_server_async()
            if self.quic_server and self.quic_server.server:
                await self.quic_server.serve_forever()
        except asyncio.CancelledError:
            bt.logging.debug("Lightning server task cancelled")
        except Exception as e:
            bt.logging.error(f"Lightning server error: {e}")
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

    @with_rate_limit(10, 60)
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

    @with_rate_limit(10, 60)
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

    @with_rate_limit(10, 60)
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
        try:
            if hasattr(circuit_store, "circuits") and circuit_store.circuits:
                circuit = circuit_store.circuits.get(synapse.model_id)
                if circuit:
                    verified_model_session = VerifiedModelSession(
                        circuit=circuit,
                        model_timeout=CIRCUIT_TIMEOUT_SECONDS,
                        should_store_proof=synapse.save,
                    )

                    result = verified_model_session.query(
                        GenericInput(
                            RequestType.POC, synapse.query_input["public_inputs"]
                        )
                    )

                    if result.is_success:
                        synapse.query_output = result.data
                        bt.logging.info(
                            f"Successfully processed proof for {synapse.model_id}"
                        )
                    else:
                        synapse.query_output = ""
                        bt.logging.warning(
                            f"Failed to process proof for {synapse.model_id}: {result.error_message}"
                        )
                else:
                    synapse.query_output = ""
                    bt.logging.warning(
                        f"Model {synapse.model_id} not found in circuit store"
                    )
            else:
                synapse.query_output = ""
                bt.logging.warning("Circuit store is empty or not initialized")

            self.log_batch.append(
                {
                    "model_id": synapse.model_id,
                    "query_output": synapse.query_output,
                    "timestamp": time.time(),
                }
            )

        except Exception as e:
            bt.logging.error(f"Error in queryZkProof: {e}")
            synapse.query_output = ""

        return synapse

    def handle_pow_request(
        self, synapse: ProofOfWeightsSynapse
    ) -> ProofOfWeightsSynapse:
        try:
            if hasattr(circuit_store, "circuits") and circuit_store.circuits:
                circuit = circuit_store.circuits.get(synapse.verification_key_hash)
                if circuit:
                    verified_model_session = VerifiedModelSession(
                        circuit=circuit,
                        model_timeout=CIRCUIT_TIMEOUT_SECONDS,
                        should_store_proof=True,
                    )

                    input_data = json.loads(synapse.inputs)

                    result = verified_model_session.query(
                        GenericInput(RequestType.POW, input_data)
                    )

                    if result.is_success:
                        synapse.proof = result.data.get("proof", "")
                        synapse.public_signals = result.data.get("public_signals", "")
                        bt.logging.info(
                            f"Successfully processed PoW for {synapse.verification_key_hash}"
                        )
                    else:
                        synapse.proof = ""
                        synapse.public_signals = ""
                        bt.logging.warning(
                            f"Failed to process PoW for {synapse.verification_key_hash}: {result.error_message}"
                        )
                else:
                    synapse.proof = ""
                    synapse.public_signals = ""
                    bt.logging.warning(
                        f"Circuit {synapse.verification_key_hash} not found in circuit store"
                    )
            else:
                synapse.proof = ""
                synapse.public_signals = ""
                bt.logging.warning("Circuit store is empty or not initialized")

        except Exception as e:
            bt.logging.error(f"Error in handle_pow_request: {e}")
            synapse.proof = ""
            synapse.public_signals = ""

        return synapse

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
