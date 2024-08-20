import argparse
import os
import traceback

import bittensor as bt
import utils.wandb_logger as wandb_logger
from _miner.miner_session import MinerSession
from utils import run_shared_preflight_checks


def get_config_from_args():
    """
    Creates the configuration from CLI args.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--netuid", type=int, default=1, help="The UID for the Omron subnet."
    )
    parser.add_argument(
        "--no-auto-update",
        default=False,
        help="Whether this miner should NOT automatically update upon new release.",
        action="store_true",
    )
    parser.add_argument(
        "--disable-blacklist",
        default=False,
        action="store_true",
        help="Disables request filtering and allows all incoming requests.",
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

    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)

    config = bt.config(parser)

    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # type: ignore
            config.wallet.name,  # type: ignore
            config.wallet.hotkey,  # type: ignore
            config.netuid,
            "miner",
        )
    )

    bt.logging(config=config, logging_dir=config.full_path)

    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)

    if config.wandb_key:
        wandb_logger.safe_login(api_key=config.wandb_key)
        bt.logging.success("Logged into WandB")
    return config


if __name__ == "__main__":
    bt.logging.info("Getting miner configuration...")
    config = get_config_from_args()

    run_shared_preflight_checks()

    try:
        bt.logging.info("Creating miner session...")
        miner_session = MinerSession(config)
        bt.logging.info("Running main loop...")
        miner_session.run()
    except Exception:
        bt.logging.error(
            f"CRITICAL: Failed to run miner session\n{traceback.format_exc()}"
        )
