"""
Safe methods for WandB logging
"""

import bittensor as bt
import psutil
import torch
import wandb

ENTITY_NAME = "inferencelabs"
PROJECT_NAME = "omron"
WANDB_ENABLED = False


def safe_login(api_key):
    """
    Attempts to log into WandB using a provided API key
    """
    try:
        bt.logging.debug("Attempting to log into WandB using provided API Key")
        wandb.login(key=api_key)
    except Exception as e:
        bt.logging.error(e)
        bt.logging.error("Failed to login to WandB. Your run will not be logged.")


def safe_init(name, wallet, metagraph, config):
    """
    Attempts to initialize WandB, and logs if unsuccessful
    """
    global WANDB_ENABLED
    if config.disable_wandb:
        bt.logging.warning("WandB logging disabled.")
        WANDB_ENABLED = False
        return
    try:
        bt.logging.debug("Attempting to initialize WandB")
        config_dict = {
            "netuid": config.netuid,
            "hotkey": wallet.hotkey.ss58_address,
            "coldkey": wallet.coldkeypub.ss58_address,
            "uid": metagraph.hotkeys.index(wallet.hotkey.ss58_address),
            "cpu_physical": psutil.cpu_count(logical=False),
            "cpu_logical": psutil.cpu_count(logical=True),
            "cpu_freq": psutil.cpu_freq().max,
            "memory": psutil.virtual_memory().total,
        }

        # Log GPU specs if available
        if torch.cuda.is_available():
            bt.logging.info("CUDA is available, attempting to get GPU specs...")
            try:
                config_dict["gpu_count"] = torch.cuda.device_count()
                config_dict["gpu_mem"] = [0] * config_dict["gpu_count"]
                config_dict["processor_count"] = [0] * config_dict["gpu_count"]

                for i in range(config_dict["gpu_count"]):
                    props = torch.cuda.get_device_properties(i)
                    config_dict["gpu_mem"][i] += props.total_memory
                    config_dict["processor_count"][i] += props.multi_processor_count
            except Exception as e:
                bt.logging.error(
                    f"Failed to retrieve or process CUDA device properties: {e}"
                )
                config_dict["gpu_count"] = 0
                config_dict["gpu_mem"] = []
                config_dict["processor_count"] = []

        # Configure destination project based on target network
        project_name = PROJECT_NAME
        if config.dev:
            project_name = PROJECT_NAME + "-development"
        elif config.subtensor.network == "test":
            project_name = PROJECT_NAME + "-testnet"

        # Disable WandB if the user has requested it
        mode = "online"
        if config.disable_wandb:
            mode = "disabled"

        # Setting logging off for miners
        if name == "Miner":
            console = "off"
        else:
            console = "auto"

        wandb.init(
            project=project_name,
            mode=mode,
            entity=ENTITY_NAME,
            anonymous="allow",
            name=name + "-" + str(wallet.hotkey.ss58_address),
            config=config_dict,
            settings=wandb.Settings(console=console),
        )
        WANDB_ENABLED = True
        bt.logging.success("Successfully configured WandB.")
    except Exception as e:
        bt.logging.warning("Failed to configure WandB. Your run will not be logged.")
        bt.logging.warning(e)


def safe_log(data):
    """
    Safely log data to WandB
    - Ignores request to log if WandB isn't configured
    - Logs to WandB if it is configured
    """

    if not WANDB_ENABLED:
        bt.logging.debug("Skipping log due to WandB logging disabled.")
        return

    try:
        bt.logging.debug("Attempting to log data to WandB")
        wandb.log(data)
    except Exception as e:
        bt.logging.debug("Failed to log to WandB.")
        bt.logging.debug(e)
