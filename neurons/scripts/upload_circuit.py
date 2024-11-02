import os
import fire
from pathlib import Path
import bittensor as bt

import requests

PINATA_URL = "https://api.pinata.cloud/pinning/pinFileToIPFS"


def upload_circuit(
    base_directory: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ),
    pinata_jwt: str = "",
    wallet_name: str = "default",
    hotkey_name: str = "default",
    wallet_path: str = None,
):
    """
    Uploads a directory to IPFS and saves the CID to a file.

    Args:
        directory (str): Path to the directory to be uploaded
    """

    wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_path)
    ss58_address = wallet.get_hotkey().ss58_address

    cid_path = os.path.join(base_directory, "neurons", "_miner", "CIRCUIT_CID")
    circuit_directory = os.path.join(base_directory, "competition_circuit")

    if not pinata_jwt:
        raise ValueError("Pinata JWT must be provided")

    if not os.path.isdir(circuit_directory):
        raise ValueError(f"Directory {circuit_directory} does not exist")

    with open(os.path.join(circuit_directory, ss58_address), "w") as f:
        pass

    pinata_headers = {"Authorization": f"Bearer {pinata_jwt}"}

    try:
        files = []
        for filename in os.listdir(circuit_directory):
            filepath = os.path.join(circuit_directory, filename)
            if os.path.isfile(filepath):
                files.append(
                    (
                        "file",
                        (
                            f"{Path(circuit_directory).name}/{filename}",
                            open(filepath, "rb"),
                        ),
                    )
                )
        response = requests.post(PINATA_URL, headers=pinata_headers, files=files)
        if response.status_code == 200:
            cid = response.json()["IpfsHash"]
            with open(cid_path, "w") as f:
                f.write(cid)
            print(f"Circuit CID {cid} saved to {cid_path} and pinned to Pinata")
        else:
            raise Exception(f"Pinata upload failed: {response.text}")
    except Exception as e:
        print(f"Error uploading to Pinata: {str(e)}")
        raise


if __name__ == "__main__":
    fire.Fire(upload_circuit)
