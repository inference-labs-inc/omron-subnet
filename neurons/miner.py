import argparse
import os

import bittensor as bt
import wandb_logger
from _miner.miner_session import MinerSession
from utils import sync_model_files
import traceback
import subprocess


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

    if config.wandb_key:
        wandb_logger.safe_login(api_key=config.wandb_key)
        bt.logging.success("Logged into WandB")
    return config


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    # Parse the configuration.
    bt.logging.info("Getting miner configuration...")
    config = get_config_from_args()

    # Sync remote model files
    try:
        sync_model_files()
    except Exception as e:
        bt.logging.error(
            "Failed to sync model files. Please run ./sync_model_files.sh to manually sync them.",
            e,
        )
        traceback.print_exc()

    # Startup TEE worker
    try:
        bt.logging.info("Starting FastChat worker in TEE...")
        import yaml
        import os

        worker_yaml_path = os.path.join(
            os.path.dirname(__file__),
            "deployment_layer",
            "tee",
            "miner",
            "worker.yaml",
        )
        with open(worker_yaml_path, "r") as file:
            compose_config = yaml.safe_load(file)

        # Update the WORKER_HOST environment variable
        compose_config["services"]["executor"]["environment"].append(
            f"WORKER_HOST={subprocess.check_output(['hostname', '-i']).decode().strip()}"
        )

        # Write the updated configuration back to the file
        with open(worker_yaml_path, "w") as file:
            yaml.dump(compose_config, file)

        docker_command = ["docker", "compose", "-f", worker_yaml_path, "up", "-d"]
        subprocess.run(docker_command, check=True)
        bt.logging.success("FastChat worker started successfully in TEE.")
    except subprocess.CalledProcessError as e:
        bt.logging.error(f"Failed to start FastChat worker in TEE: {e}")
    except Exception as e:
        bt.logging.error(
            f"An unexpected error occurred while starting FastChat worker: {e}"
        )

    # Run the main function.
    try:
        bt.logging.info("Creating miner session...")
        miner_session = MinerSession(config)
        bt.logging.info("Running main loop...")
        miner_session.run()
    except Exception as e:
        bt.logging.error("error", e)
