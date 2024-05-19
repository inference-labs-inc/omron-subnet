import argparse
import json
import os
import random
import time
import traceback

import bittensor as bt
import protocol
import torch
import wandb_logger
from _validator.validator_session import ValidatorSession


# This function is responsible for setting up and parsing command-line arguments.
def get_config_from_args():
    """
    This function sets up and parses command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Adds override arguments for network and netuid.
    parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
    parser.add_argument("--auto-update", default=True, help="Auto update.")
    parser.add_argument(
        "--blocks_per_epoch",
        type=int,
        default=50,
        help="Number of blocks to wait before setting weights",
    )
    parser.add_argument(
        "--wandb-key", type=str, default="", help="A https://wandb.ai API key"
    )

    parser.add_argument(
        "--disable-wandb",
        default=False,
        help="Whether to disable WandB logging.",
        action="store_true",
    )
    parser.add_argument(
        "--dev",
        default=False,
        help="Whether to run the miner in development mode for internal testing.",
        action="store_true",
    )
    
    parser.add_argument(
        "--validator_batch_size",
        type=int,
        default=64,
        help="Number of requests to miners performed at once."
    )

    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)

    # Parse the config (will take command-line arguments if provided)
    # To print help message, run python3 neurons/validator.py --help
    config = bt.config(parser)

    # Logging is crucial for monitoring and debugging purposes.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "validator",
        )
    )
    # Ensure the logging directory exists.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)

    # Set up logging with the provided configuration and directory.
    bt.logging(config=config, logging_dir=config.full_path)

    if config.wandb_key:
        wandb_logger.safe_login(api_key=config.wandb_key)
        bt.logging.success("Logged into WandB")

    # Return the parsed config.
    return config


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    # Parse the configuration.
    config = get_config_from_args()

    # Run the main function.
    with ValidatorSession(config) as validator_session:
        validator_session.run()
