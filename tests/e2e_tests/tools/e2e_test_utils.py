import os

from bittensor import Wallet, Keypair

import bittensor

template_path = os.getcwd() + "/neurons/"
templates_repo = "templates repository"


def setup_wallet(uri: str) -> tuple[Keypair, bittensor.Wallet]:
    """
    Sets up a wallet using the provided URI.

    This function creates a keypair from the given URI and initializes a wallet
    at a temporary path. It sets the coldkey, coldkeypub, and hotkey for the wallet
    using the generated keypair.

    Side Effects:
        - Creates a wallet in a temporary directory.
        - Sets keys in the wallet without encryption and with overwriting enabled.
    """
    keypair = Keypair.create_from_uri(uri)
    wallet_path = f"/tmp/btcli-e2e-wallet-{uri.strip('/')}"
    wallet = bittensor.Wallet(path=wallet_path)
    wallet.set_coldkey(keypair=keypair, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(keypair=keypair, encrypt=False, overwrite=True)
    wallet.set_hotkey(keypair=keypair, encrypt=False, overwrite=True)
    return keypair, wallet