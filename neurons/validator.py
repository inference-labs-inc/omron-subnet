import argparse
import os
import sys
import traceback

import bittensor as bt
from constants import (
    ONCHAIN_PROOF_OF_WEIGHTS_ENABLED,
    PROOF_OF_WEIGHTS_INTERVAL,
    WHITELISTED_PUBLIC_KEYS,
)

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
        default=bool(os.getenv("OMRON_NO_AUTO_UPDATE", False)),
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
        "--disable-statistic-logging",
        default=False,
        help="Whether to disable statistic logging.",
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

    parser.add_argument(
        "--ignore-external-requests",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to ignore external requests.",
    )

    parser.add_argument(
        "--whitelisted-public-keys",
        type=str,
        nargs="*",
        dest="alist",
        default=WHITELISTED_PUBLIC_KEYS,
        help="Comma separated list of public keys to whitelist for external requests.",
    )

    parser.add_argument(
        "--external-api-host",
        type=str,
        default="0.0.0.0",
        help="The host for the external API.",
    )

    parser.add_argument(
        "--external-api-port",
        type=int,
        default=8443,
        help="The port for the external API.",
    )

    parser.add_argument(
        "--external-api-workers",
        type=int,
        default=1,
        help="The number of workers for the external API.",
    )

    parser.add_argument(
        "--do-not-verify-external-signatures",
        default=False,
        action="store_true",
        help=(
            "External PoW requests are signed by validator's (sender's) wallet. "
            "By default we verify is the wallet legitimate. "
            "You can disable this check with the flag."
        ),
    )

    parser.add_argument(
        "--localnet",
        action="store_true",
        default=False,
        help="Whether to run the validator in localnet mode.",
    )

    parser.add_argument(
        "--certificate-path",
        type=str,
        default=None,
        help="A custom path to a directory containing a public and private SSL certificate. "
        "(cert.pem and key.pem) "
        "Please note that this should not be used unless you have issued your own certificate. "
        "Omron will issue a certificate for you by default.",
    )

    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
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
            config.wallet.name = "validator"
        if config.subtensor.network == "finney":
            config.subtensor.network = "local"
        config.eth_wallet = (
            config.eth_wallet if config.eth_wallet is not None else "0x001"
        )
        config.timeout = config.timeout if config.timeout is None else 120
        config.disable_wandb = True
        config.verbose = config.verbose if config.verbose is None else True
        config.disable_blacklist = (
            config.disable_blacklist if config.disable_blacklist is None else True
        )
        config.external_api_workers = config.external_api_workers or 1
        config.external_api_port = config.external_api_port or 8443

    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # type: ignore
            config.wallet.name,  # type: ignore
            config.wallet.hotkey,  # type: ignore
            config.netuid,
            "validator",
        )
    )

    if not config.certificate_path:
        config.certificate_path = os.path.join(config.full_path, "cert")

    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)

    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.enable_info()

    if config.wandb_key:
        wandb_logger.safe_login(api_key=config.wandb_key)
        bt.logging.success("Logged into WandB")

    return config


if __name__ == "__main__":
    bt.logging.info("Getting validator configuration...")
    config = get_config_from_args()

    config.external_model_dir = os.path.join(
        os.path.dirname(config.full_path), "deployment_layer"
    )
    os.environ["OMRON_EXTERNAL_MODEL_DIR"] = config.external_model_dir
    os.environ["EZKL_REPO_PATH"] = os.path.join(
        os.path.dirname(config.full_path), "ezkl"
    )

    run_shared_preflight_checks(config.external_model_dir)

    if os.getenv("OMRON_DOCKER_BUILD", False):
        bt.logging.info("Docker build steps complete. Exiting.")
        sys.exit(0)

    try:
        # Initialize the circuit store and load external models
        from deployment_layer.circuit_store import circuit_store

        circuit_store.load_circuits(config.external_model_dir)

        bt.logging.info("Creating validator session...")
        validator_session = ValidatorSession(config)
        bt.logging.info("Running main loop...")
        validator_session.run()
    except Exception as e:
        bt.logging.error("Critical error while attempting to run validator: ", e)
        traceback.print_exc()
