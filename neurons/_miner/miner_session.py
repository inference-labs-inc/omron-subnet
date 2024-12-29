# from __future__ import annotations
import json
import time
import traceback
from typing import Tuple, Union

import bittensor as bt
import websocket

import cli_parser
from _validator.models.request_type import RequestType
from constants import (
    SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
    STEAK,
    VALIDATOR_REQUEST_TIMEOUT_SECONDS,
    VALIDATOR_STAKE_THRESHOLD,
)
from deployment_layer.circuit_store import circuit_store
from execution_layer.generic_input import GenericInput
from execution_layer.verified_model_session import VerifiedModelSession
from protocol import ProofOfWeightsSynapse, QueryForProofAggregation, QueryZkProof
from utils import AutoUpdate, clean_temp_files, wandb_logger


class MinerSession:

    axon: Union[bt.axon, None] = None

    def __init__(self):
        self.configure()
        self.check_register(should_exit=True)
        self.auto_update = AutoUpdate()
        self.log_batch = []
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

        # Attach determines which functions are called when a request is received.
        bt.logging.info("Attaching forward functions to axon...")
        axon.attach(forward_fn=self.queryZkProof, blacklist_fn=self.proof_blacklist)
        axon.attach(
            forward_fn=self.handle_pow_request,
            blacklist_fn=self.pow_blacklist,
        )
        bt.logging.info("Attached forward functions to axon")

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip has changed.
        bt.logging.info(
            f"Serving axon on network: {self.subtensor.chain_endpoint} with netuid: {cli_parser.config.netuid}"
        )
        axon.serve(netuid=cli_parser.config.netuid, subtensor=self.subtensor)
        bt.logging.info(
            f"Served axon on network: {self.subtensor.chain_endpoint} with netuid: {cli_parser.config.netuid}"
        )

        # Start the miner's axon, making it active on the network.
        bt.logging.info(f"Starting axon server: {axon.info()}")
        axon.start()
        bt.logging.info(f"Started axon server: {axon.info()}")

        self.axon = axon

    def run(self):
        """
        Keep the miner alive.
        This loop maintains the miner's operations until intentionally stopped.
        """
        bt.logging.info("Starting miner...")
        self.start_axon()

        step = 0

        while True:
            step += 1
            try:
                if step % 10 == 0:
                    if not cli_parser.config.no_auto_update:
                        self.auto_update.try_update()
                    else:
                        bt.logging.info(
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
                    try:
                        self.metagraph = self.subtensor.metagraph(
                            cli_parser.config.netuid
                        )
                        bt.logging.info(
                            f"Step:{step} | "
                            f"Block:{self.metagraph.block.item()} | "
                            f"Stake:{self.metagraph.S[self.subnet_uid]} | "
                            f"Rank:{self.metagraph.R[self.subnet_uid]} | "
                            f"Trust:{self.metagraph.T[self.subnet_uid]} | "
                            f"Consensus:{self.metagraph.C[self.subnet_uid]} | "
                            f"Incentive:{self.metagraph.I[self.subnet_uid]} | "
                            f"Emission:{self.metagraph.E[self.subnet_uid]}"
                        )
                    except Exception:
                        bt.logging.warning(
                            f"Failed to sync metagraph: {traceback.format_exc()}"
                        )

                time.sleep(1)

            # If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                bt.logging.success("Miner killed via keyboard interrupt.")
                clean_temp_files()
                break
            # In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception:
                bt.logging.error(traceback.format_exc())
                continue

    def check_register(self, should_exit=False):
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour miner: {self.wallet} is not registered to the network: {self.subtensor} \n"
                "Run btcli register and try again."
            )
            if should_exit:
                exit()
            self.subnet_uid = None
        else:
            # Each miner gets a unique identity (UID) in the network for differentiation.
            subnet_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(f"Running miner on uid: {subnet_uid}")
            self.subnet_uid = subnet_uid

    def configure(self):
        # === Configure Bittensor objects ====
        self.wallet = bt.wallet(config=cli_parser.config)
        self.subtensor = bt.subtensor(config=cli_parser.config)
        self.metagraph = self.subtensor.metagraph(cli_parser.config.netuid)
        wandb_logger.safe_init("Miner", self.wallet, self.metagraph, cli_parser.config)

    def proof_blacklist(self, synapse: QueryZkProof) -> Tuple[bool, str]:
        """
        Blacklist method for the proof generation endpoint
        """
        return self._blacklist(synapse)

    def aggregation_blacklist(
        self, synapse: QueryForProofAggregation
    ) -> Tuple[bool, str]:
        """
        Blacklist method for the aggregation endpoint
        """
        return self._blacklist(synapse)

    def pow_blacklist(self, synapse: ProofOfWeightsSynapse) -> Tuple[bool, str]:
        """
        Blacklist method for the proof generation endpoint
        """
        return self._blacklist(synapse)

    def _blacklist(
        self,
        synapse: Union[QueryZkProof, QueryForProofAggregation, ProofOfWeightsSynapse],
    ) -> Tuple[bool, str]:
        """
        Filters requests if any of the following conditions are met:
        - Requesting hotkey is not registered
        - Requesting UID's stake is below 1k
        - Requesting UID does not have a validator permit

        Does not filter if the --disable-blacklist flag has been set.

        synapse: The request synapse object
        returns: (is_blacklisted, reason)
        """
        try:
            if cli_parser.config.disable_blacklist:
                bt.logging.trace("Blacklist disabled, allowing request.")
                return False, "Allowed"

            if synapse.dendrite.hotkey not in self.metagraph.hotkeys:  # type: ignore
                return True, "Hotkey is not registered"

            requesting_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)  # type: ignore
            stake = self.metagraph.S[requesting_uid].item()

            try:
                bt.logging.info(
                    f"Request by: {synapse.dendrite.hotkey} | UID: {requesting_uid} "  # type: ignore
                    f"| Stake: {stake} {STEAK}"
                )
            except UnicodeEncodeError:
                bt.logging.info(
                    f"Request by: {synapse.dendrite.hotkey} | UID: {requesting_uid} | Stake: {stake}"  # type: ignore
                )

            if stake < VALIDATOR_STAKE_THRESHOLD:
                return True, "Stake below minimum"

            validator_permit = self.metagraph.validator_permit[requesting_uid].item()
            if not validator_permit:
                return True, "Requesting UID has no validator permit"

            bt.logging.trace(f"Allowing request from UID: {requesting_uid}")
            return False, "Allowed"

        except Exception as e:
            bt.logging.error(f"Error during blacklist {e}")
            return True, "An error occurred while filtering the request"

    def queryZkProof(self, synapse: QueryZkProof) -> QueryZkProof:
        """
        This function run proof generation of the model (with its output as well)
        """
        time_in = time.time()
        bt.logging.debug("Received request from validator")
        bt.logging.debug(f"Input data: {synapse.query_input} \n")

        if not synapse.query_input or not synapse.query_input.get(
            "public_inputs", None
        ):
            bt.logging.error("Received empty query input")
            synapse.query_output = "Empty query input"
            return synapse

        model_id = synapse.query_input.get("model_id", SINGLE_PROOF_OF_WEIGHTS_MODEL_ID)
        public_inputs = synapse.query_input["public_inputs"]

        # Run inputs through the model and generate a proof.
        try:
            circuit = circuit_store.get_circuit(str(model_id))
            if not circuit:
                raise ValueError(
                    f"Circuit {model_id} not found. This indicates a missing deployment layer folder or invalid request"
                )
            bt.logging.info(f"Running proof generation for {circuit}")
            model_session = VerifiedModelSession(
                GenericInput(RequestType.RWR, public_inputs), circuit
            )
            bt.logging.debug("Model session created successfully")
            proof, public, proof_time = model_session.gen_proof()
            if isinstance(proof, bytes):
                proof = proof.hex()

            synapse.query_output = json.dumps(
                {
                    "proof": proof,
                    "public_signals": public,
                }
            )
            bt.logging.trace(f"Proof: {synapse.query_output}, Time: {proof_time}")
            model_session.end()
            try:
                bt.logging.info(f"✅ Proof completed for {circuit}\n")
            except UnicodeEncodeError:
                bt.logging.info(f"Proof completed for {circuit}\n")
        except Exception as e:
            synapse.query_output = "An error occurred"
            bt.logging.error(f"An error occurred while generating proven output\n{e}")
            traceback.print_exc()
            proof_time = time.time() - time_in

        time_out = time.time()
        delta_t = time_out - time_in
        bt.logging.info(
            f"Total response time {delta_t}s. Proof time: {proof_time}s. "
            f"Overhead time: {delta_t - proof_time}s."
        )
        self.log_batch.append(
            {
                str(model_id): {
                    "proof_time": proof_time,
                    "overhead_time": delta_t - proof_time,
                    "total_response_time": delta_t,
                }
            }
        )

        if delta_t > VALIDATOR_REQUEST_TIMEOUT_SECONDS:
            bt.logging.error(
                "Response time is greater than validator timeout. "
                "This indicates your hardware is not processing validator's requests in time."
            )
        return synapse

    def handle_pow_request(
        self, synapse: ProofOfWeightsSynapse
    ) -> ProofOfWeightsSynapse:
        """
        Handles a proof of weights request
        """
        time_in = time.time()
        bt.logging.debug("Received proof of weights request from validator")
        bt.logging.debug(f"Input data: {synapse.inputs} \n")

        if not synapse.inputs:
            bt.logging.error("Received empty input for proof of weights")
            return synapse

        try:
            circuit = circuit_store.get_circuit(str(synapse.verification_key_hash))

            if not circuit:
                raise ValueError(
                    f"Circuit {synapse.verification_key_hash} not found. "
                    "This indicates a missing deployment layer folder or invalid request"
                )
            bt.logging.info(f"Running proof generation for {circuit}")
            model_session = VerifiedModelSession(
                GenericInput(RequestType.RWR, synapse.inputs), circuit
            )

            bt.logging.debug("Model session created successfully")
            proof, public, proof_time = model_session.gen_proof()
            model_session.end()

            synapse.proof = proof.hex() if isinstance(proof, bytes) else proof
            synapse.public_signals = public
            bt.logging.info(f"✅ Proof of weights completed for {circuit}\n")
        except Exception as e:
            bt.logging.error(
                f"An error occurred while generating proof of weights\n{e}"
            )
            traceback.print_exc()
            proof_time = time.time() - time_in

        time_out = time.time()
        delta_t = time_out - time_in
        bt.logging.info(
            f"Total response time {delta_t}s. Proof time: {proof_time}s. "
            f"Overhead time: {delta_t - proof_time}s."
        )
        self.log_batch.append(
            {
                str(synapse.verification_key_hash): {
                    "proof_time": proof_time,
                    "overhead_time": delta_t - proof_time,
                    "total_response_time": delta_t,
                }
            }
        )

        if delta_t > VALIDATOR_REQUEST_TIMEOUT_SECONDS:
            bt.logging.error(
                "Response time is greater than validator timeout. "
                "This indicates your hardware is not processing validator's requests in time."
            )
        return synapse

    def aggregateProof(
        self, synapse: QueryForProofAggregation
    ) -> QueryForProofAggregation:
        """
        Generates an aggregate proof for the provided proofs.
        """
        raise NotImplementedError("Proof aggregation not supported at this time.")
