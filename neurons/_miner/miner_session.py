import time
import traceback

import bittensor as bt
import protocol
from execution_layer.VerifiedModelSession import VerifiedModelSession
from utils import AutoUpdate


class MinerSession:
    def __init__(self, config):
        self.config = config
        self.configure()
        self.check_register()
        self.auto_update = AutoUpdate()
        self.axon = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def unpack_bt_objects(self):
        wallet = self.wallet
        metagraph = self.metagraph
        subtensor = self.subtensor
        return wallet, metagraph, subtensor

    def start_axon(self):
        wallet, metagraph, subtensor = self.unpack_bt_objects()
        bt.logging.info(
            f"Starting axon on port: {self.config.axon.port}, using external IP: {self.config.axon.external_ip}"
        )

        axon = bt.axon(wallet=wallet, config=self.config)
        bt.logging.info(f"Axon created: {axon.info()}")

        # Attach determines which functions are called when a request is received.
        bt.logging.info("Attaching forward function to axon...")
        axon.attach(forward_fn=self.queryZkProof)
        bt.logging.info("Attached forward function to axon")

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip has changed.
        bt.logging.info(
            f"Serving axon on network: {subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        axon.serve(netuid=self.config.netuid, subtensor=subtensor)
        bt.logging.info(
            f"Served axon on network: {subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        # Start the miner's axon, making it active on the network.
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        axon.start()
        bt.logging.info(f"Started axon server on port: {self.config.axon.port}")

        self.axon = axon

    def run(self):
        """Keep the miner alive. This loop maintains the miner's operations until intentionally stopped."""

        bt.logging.info("Starting miner...")
        _, metagraph, subtensor = self.unpack_bt_objects()

        self.start_axon()

        step = 0

        while True:
            if step % 10 == 0 and self.config.auto_update == True:
                self.auto_update.try_update()
            try:
                if step % 5 == 0:
                    metagraph = subtensor.metagraph(self.config.netuid)
                    log = (
                        f"Step:{step} | "
                        f"Block:{metagraph.block.item()} | "
                        f"Stake:{metagraph.S[self.subnet_uid]} | "
                        f"Rank:{metagraph.R[self.subnet_uid]} | "
                        f"Trust:{metagraph.T[self.subnet_uid]} | "
                        f"Consensus:{metagraph.C[self.subnet_uid] } | "
                        f"Incentive:{metagraph.I[self.subnet_uid]} | "
                        f"Emission:{metagraph.E[self.subnet_uid]}"
                    )
                    bt.logging.info(log)
                step += 1
                time.sleep(1)

            # If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                bt.logging.success("Miner killed via keyboard interrupt.")
                break
            # In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception as e:
                bt.logging.error(traceback.format_exc())
                continue

    def check_register(self):
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour miner: {self.wallet} if not registered to the network: {self.subtensor} \nRun btcli register and try again."
            )
            exit()
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
        self.sync_metagraph()

    def sync_metagraph(self):
        self.metagraph.sync(subtensor=self.subtensor)

    def queryZkProof(self, synapse: protocol.QueryZkProof) -> protocol.QueryZkProof:
        """
        This function run proof generation of the model (with its output as well)
        """
        time_in = time.time()
        bt.logging.debug("Received request from validator")
        bt.logging.info(f"Input data: {synapse.query_input} \n")
        if synapse.query_input is not None:
            model_id = synapse.query_input["model_id"]
            public_inputs = synapse.query_input["public_inputs"]
        else:
            bt.logging.error("Received empty query input")

        # Run inputs through the model and generate a proof.
        try:
            model_session = VerifiedModelSession(public_inputs)
            bt.logging.debug("Model session created successfully")
            synapse.query_output = model_session.gen_proof()
            model_session.end()
        except Exception as e:
            synapse.query_output = "An error occured"

            bt.logging.error("An error occurred while generating proven output", e)

        bt.logging.info("Proof completed \n")
        time_out = time.time()
        delta_t = time_out - time_in
        bt.logging.info(f"Request to Response time {delta_t}s")
        if delta_t > 300:
            bt.logging.error(
                "Turnaround time is greater than validator timeout. This indicates your hardware is not processing validator's requests in time."
            )
        return synapse
