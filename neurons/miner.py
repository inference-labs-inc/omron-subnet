import argparse
import json
import os
import random
import time
import traceback

import bittensor as bt
import protocol
import torch
from _miner.miner_session import MinerSession

# This function is responsible for setting up and parsing command-line arguments.


def get_config_from_args():
    """
    This function initializes the necessary command-line arguments.
    Using command-line arguments allows users to customize various miner settings.
    """
    parser = argparse.ArgumentParser()
    # Adds override arguments for network and netuid.
    parser.add_argument(
        "--netuid", type=int, default=1, help="The UID for the Omron subnet."
    )
    parser.add_argument(
        "--auto-update",
        default=True,
        help="Whether this miner should automatically update upon new release.",
    )
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Adds axon specific arguments i.e. --axon.port ...
    bt.axon.add_args(parser)
    # Activating the parser to read any command-line inputs.
    # To print help message, run python3 neurons/miner.py --help
    config = bt.config(parser)

    # Set up logging directory
    # Logging captures events for diagnosis or understanding miner's behavior.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "miner",
        )
    )
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)

    return config


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    # Parse the configuration.
    bt.logging.info("Getting miner configuration...")
    config = get_config_from_args()
    # Run the main function.
    try:
        bt.logging.info("Creating miner session...")
        miner_session = MinerSession(config)
        bt.logging.info("Running main loop...")
        miner_session.run()
    except Exception as e:
        bt.logging.error("error", e)
