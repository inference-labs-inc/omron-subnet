import sys
import bittensor as bt
from constants import DEFAULT_NETUID

from utils import wandb_logger


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
    """

    def __init__(self, config: bt.config):
        """
        Initialize the ValidatorConfig object.

        Args:
            config (bt.config): The Bittensor configuration object.
        """
        self.config = config
        self.subnet_uid = int(
            self.config.netuid if self.config.netuid else DEFAULT_NETUID
        )
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(self.subnet_uid)
        self.user_uid = int(
            self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        )

        # Initialize wandb logger
        wandb_logger.safe_init(
            "Validator",
            self.wallet,
            self.metagraph,
            self.config,
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
