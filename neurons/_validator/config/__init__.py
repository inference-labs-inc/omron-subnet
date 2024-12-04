import sys
import bittensor as bt
from constants import DEFAULT_NETUID

from utils import wandb_logger


class ApiConfig:
    """
    Configuration class for the API.

    Attributes:
        enabled (bool): Whether the API is enabled.
        host (str): The host for the API.
        port (int): The port for the API.
        workers (int): The number of workers for the API.
    """

    def __init__(self, config: bt.config):
        self.enabled = not config.ignore_external_requests
        self.host = config.external_api_host
        self.port = config.external_api_port
        self.workers = config.external_api_workers
        self.verify_external_signatures = not config.do_not_verify_external_signatures


class ValidatorConfig:
    """
    Configuration class for the Validator.

    This class initializes and manages the configuration settings for the Omron validator.

    Attributes:
        config (bt.config): The Bittensor configuration object.
        subnet_uid (int): The unique identifier for the subnet.
        wallet (bt.wallet): The Bittensor wallet object.
        subtensor (bt.subtensor): The Bittensor subtensor object.
        dendrite (bt.dendrite): The Bittensor dendrite object.
        metagraph (bt.metagraph): The Bittensor metagraph object.
        user_uid (int): The unique identifier for the validator within the subnet's metagraph.
        api_enabled (bool): Whether the API is enabled.
    """

    def __init__(self, config: bt.config):
        """
        Initialize the ValidatorConfig object.

        Args:
            config (bt.config): The Bittensor configuration object.
        """
        self.bt_config = config
        self.subnet_uid = int(
            self.bt_config.netuid if self.bt_config.netuid else DEFAULT_NETUID
        )
        self.wallet = bt.wallet(config=self.bt_config)
        self.subtensor = bt.subtensor(config=self.bt_config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(self.subnet_uid)
        self.user_uid = int(
            self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        )
        self.api = ApiConfig(self.bt_config)

        # Initialize wandb logger
        wandb_logger.safe_init(
            "Validator",
            self.wallet,
            self.metagraph,
            self.bt_config,
        )

        # Get the full_path used for logs from get_config_from_args() to be used for scores.pt path
        try:
            self.full_path = config.full_path
        except:
            config.full_path = os.path.expanduser(
                "{}/{}/{}/netuid{}/{}".format(
                    config.logging.logging_dir,  # type: ignore
                    config.wallet.name,  # type: ignore
                    config.wallet.hotkey,  # type: ignore
                    config.netuid,
                    "validator",
                )
            )

    def check_register(self):
        """
        Check if the validator is registered on the subnet.

        This method verifies if the validator's hotkey is registered in the metagraph.
        If not registered, it logs an error and exits.
        If registered, it sets the user_uid and logs it.

        Raises:
            SystemExit: If the validator is not registered on the network.
        """
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour validator: {self.wallet} is not registered to the chain: "
                f"{self.subtensor} \nRun btcli register and try again."
            )
            sys.exit(1)
        else:
            uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(f"Running validator on uid: {uid}")
            self.user_uid = uid
