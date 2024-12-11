import argparse
import os
import sys
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
        default=bool(os.getenv("OMRON_NO_AUTO_UPDATE", False)),
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
    parser.add_argument(
        "--localnet",
        action="store_true",
        default=False,
        help="Whether to run the miner in localnet mode.",
    )

    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)

    config = bt.config(parser)

    if config.localnet:
        # quick localnet configuration set up for testing
        if (
            config.subtensor.chain_endpoint
            == "wss://entrypoint-finney.opentensor.ai:443"
        ):
            # in case of default value, change to localnet
            config.subtensor.chain_endpoint = "ws://127.0.0.1:9946"
        if config.wallet.name == "default":
            config.wallet.name = "miner"
        if config.subtensor.network == "finney":
            config.subtensor.network = "local"
        config.eth_wallet = (
            config.eth_wallet if config.eth_wallet is not None else "0x002"
        )
        if not config.axon:
            config.axon = bt.config()
            config.axon.ip = "127.0.0.1"
            config.axon.external_ip = "127.0.0.1"
        config.timeout = config.timeout if config.timeout is None else 120
        config.disable_wandb = True
        config.verbose = config.verbose if config.verbose is None else True
        config.disable_blacklist = (
            config.disable_blacklist if config.disable_blacklist is None else True
        )
        config.max_workers = config.max_workers or 1

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

    external_model_dir = os.path.join(
        os.path.dirname(config.full_path), "deployment_layer"
    )
    os.environ["EZKL_REPO_PATH"] = os.path.join(
        os.path.dirname(config.full_path), "ezkl"
    )

    run_shared_preflight_checks(external_model_dir)

    if os.getenv("OMRON_DOCKER_BUILD", False):
        bt.logging.info("Docker build steps complete. Exiting.")
        sys.exit(0)

    try:
        # Initialize the circuit store and load external models
        from deployment_layer.circuit_store import circuit_store

        circuit_store.load_circuits(external_model_dir)

        bt.logging.info("Creating miner session...")
        miner_session = MinerSession(config)
        bt.logging.info("Running main loop...")
        miner_session.run()
    except Exception:
        bt.logging.error(
            f"CRITICAL: Failed to run miner session\n{traceback.format_exc()}"
        )
