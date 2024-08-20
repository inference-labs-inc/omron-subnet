import argparse
import os

import bittensor as bt
from constants import ONCHAIN_PROOF_OF_WEIGHTS_ENABLED, PROOF_OF_WEIGHTS_INTERVAL

from utils import wandb_logger
from _validator.validator_session import ValidatorSession
from utils import run_shared_preflight_checks


def get_config_from_args():
    """
    Parses CLI arguments into bt configuration
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--netuid", type=int, default=1, help="The uid of the subnet.")

    parser.add_argument(
        "--no-auto-update",
        default=False,
        action="store_true",
        help="Disable auto update.",
    )
    parser.add_argument(
        "--blocks_per_epoch",
        type=int,
        default=100,
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
        "--enable-pow",
        default=ONCHAIN_PROOF_OF_WEIGHTS_ENABLED,
        action="store_true",
        help="Whether proof of weights is enabled",
    )

    parser.add_argument(
        "--pow-target-interval",
        type=int,
        default=PROOF_OF_WEIGHTS_INTERVAL,
        help="The target interval for committing proof of weights to the chain",
    )

    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config = bt.config(parser)

    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # type: ignore
            config.wallet.name,  # type: ignore
            config.wallet.hotkey,  # type: ignore
            config.netuid,
            "validator",
        )
    )

    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)

    bt.logging(config=config, logging_dir=config.full_path)

    if config.wandb_key:
        wandb_logger.safe_login(api_key=config.wandb_key)
        bt.logging.success("Logged into WandB")

    return config


if __name__ == "__main__":
    configuration = get_config_from_args()
    run_shared_preflight_checks()

    try:
        bt.logging.info("Creating validator session...")
        validator_session = ValidatorSession(configuration)
        bt.logging.info("Running main loop...")
        validator_session.run()
    except Exception as e:
        bt.logging.error("Critical error while attempting to run validator: ", e)
