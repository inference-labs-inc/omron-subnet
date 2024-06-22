import time
import traceback
from typing import Tuple, Union

import bittensor as bt
import protocol
import wandb_logger
import websocket
from execution_layer.VerifiedModelSession import VerifiedModelSession
from utils import AutoUpdate, clean_temp_files


class MinerSession:
    def __init__(self, config):
        self.config = config
        self.configure()
        self.check_register(should_exit=True)
        self.auto_update = AutoUpdate()
        self.axon = None
        self.log_batch = []
        if self.config.disable_blacklist:
            bt.logging.warning(
                "Blacklist disabled, allowing all requests. Consider enabling to filter requests."
            )
        websocket.setdefaulttimeout(30)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def start_axon(self):
        bt.logging.info(
            "Starting axon. Custom arguments include the following.\n"
            "Note that any null values will fallback to defaults, "
            f"which are usually sufficient. {self.config.axon}"
        )

        axon = bt.axon(wallet=self.wallet, config=self.config)
        bt.logging.info(f"Axon created: {axon.info()}")

        # Attach determines which functions are called when a request is received.
        bt.logging.info("Attaching forward functions to axon...")
        axon.attach(forward_fn=self.queryZkProof, blacklist_fn=self.proof_blacklist)
        axon.attach(
            forward_fn=self.aggregateProof, blacklist_fn=self.aggregation_blacklist
        )
        bt.logging.info("Attached forward functions to axon")

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip has changed.
        bt.logging.info(
            f"Serving axon on network: {self.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        bt.logging.info(
            f"Served axon on network: {self.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
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
            try:
                if step % 10 == 0:
                    if not self.config.no_auto_update:
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

                if step % 5 == 0 and self.subnet_uid:
                    self.metagraph = self.subtensor.metagraph(self.config.netuid)
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
                step += 1
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
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        wandb_logger.safe_init("Miner", self.wallet, self.metagraph, self.config)

    def proof_blacklist(self, synapse: protocol.QueryZkProof) -> Tuple[bool, str]:
        """
        Blacklist method for the proof generation endpoint
        """
        return self._blacklist(synapse)

    def aggregation_blacklist(
        self, synapse: protocol.QueryForProofAggregation
    ) -> Tuple[bool, str]:
        """
        Blacklist method for the aggregation endpoint
        """
        return self._blacklist(synapse)

    def _blacklist(
        self, synapse: Union[protocol.QueryZkProof, protocol.QueryForProofAggregation]
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
            if self.config.disable_blacklist:
                bt.logging.trace("Blacklist disabled, allowing request.")
                return False, "Allowed"

            if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
                return True, "Hotkey is not registered"

            requesting_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            stake = self.metagraph.S[requesting_uid].item()

            bt.logging.info(f"Requesting UID: {requesting_uid} | Stake at UID: {stake}")
            if stake < 1024:
                return True, "Stake below minimum"

            validator_permit = self.metagraph.validator_permit[requesting_uid].item()
            if not validator_permit:
                return True, "Requesting UID has no validator permit"

            bt.logging.trace(f"Allowing request from UID: {requesting_uid}")
            return False, "Allowed"

        except Exception as e:
            bt.logging.error(f"Error during blacklist {e}")
            return True, "An error occurred while filtering the request"

    def queryZkProof(self, synapse: protocol.QueryZkProof) -> protocol.QueryZkProof:
        """
        This function run proof generation of the model (with its output as well)
        """
        time_in = time.time()
        bt.logging.debug("Received request from validator")
        bt.logging.info(f"Input data: {synapse.query_input} \n")

        if not synapse.query_input or not synapse.query_input.get(
            "public_inputs", None
        ):
            bt.logging.error("Received empty query input")
            synapse.query_output = "Empty query input"
            return synapse

        model_id = synapse.query_input.get("model_id", [0])
        public_inputs = synapse.query_input["public_inputs"]
        if model_id == [0]:
            public_inputs = [public_inputs]

        # Run inputs through the model and generate a proof.
        try:
            model_session = VerifiedModelSession(public_inputs, model_id)
            bt.logging.debug("Model session created successfully")
            synapse.query_output, proof_time = model_session.gen_proof()
            model_session.end()
            try:
                bt.logging.info("✅ Proof completed \n")
            except UnicodeEncodeError:
                bt.logging.info("Proof completed \n")
        except Exception as e:
            synapse.query_output = "An error occurred"
            bt.logging.error(f"An error occurred while generating proven output\n{e}")
            proof_time = time.time() - time_in

        time_out = time.time()
        delta_t = time_out - time_in
        bt.logging.info(
            f"Total response time {delta_t}s. Proof time: {proof_time}s. "
            f"Overhead time: {delta_t - proof_time}s."
        )
        self.log_batch.append(
            {
                str(model_id[0]): {
                    "proof_time": proof_time,
                    "overhead_time": delta_t - proof_time,
                    "total_response_time": delta_t,
                }
            }
        )

        if delta_t > 300:
            bt.logging.error(
                "Response time is greater than validator timeout. "
                "This indicates your hardware is not processing validator's requests in time."
            )
        return synapse

    def aggregateProof(
        self, synapse: protocol.QueryForProofAggregation
    ) -> protocol.QueryForProofAggregation:
        """
        Generates an aggregate proof for the provided proofs.
        """
        time_in = time.time()
        bt.logging.debug(f"Aggregation input: {synapse.proofs} \n")
        bt.logging.info(
            f"Received proof aggregation request with {len(synapse.proofs)}"
        )

        if not synapse.proofs or not synapse.model_id:
            bt.logging.error(
                "Received proof aggregation request with no proofs or model_id"
            )
            synapse.aggregation_proof = "Missing critical data"
            return synapse
        aggregation_time = 0

        # Run proofs through the aggregate circuit
        try:
            model_session = VerifiedModelSession(synapse.proofs, synapse.model_id)
            bt.logging.debug("Model session created successfully")
            synapse.aggregation_proof, aggregation_time = model_session.aggregate_proof(
                synapse.proofs
            )
            model_session.end()
            try:
                bt.logging.info("✅ Aggregation completed \n")
            except UnicodeEncodeError:
                bt.logging.info("Aggregation completed \n")
        except Exception as e:
            synapse.aggregation_proof = "An error occurred"
            bt.logging.error(f"An error occurred while aggregating proofs\n{e}")

        time_out = time.time()
        delta_t = time_out - time_in
        overhead_time = delta_t - aggregation_time
        bt.logging.info(
            f"Total response time {delta_t}s. Aggregation time: {aggregation_time}s. "
            f"Overhead time: {overhead_time}s."
        )
        self.log_batch.append(
            {
                "aggregation_time": aggregation_time,
                "overhead_time": overhead_time,
                "total_response_time": delta_t,
            }
        )

        if delta_t > 300:
            bt.logging.error(
                "Response time is greater than validator timeout. "
                "This indicates your hardware is not processing validator's requests in time."
            )
        return synapse
