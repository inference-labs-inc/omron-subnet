"""

Circuit handler is responsible for handling circuits submitted by miners for competitions.
It covers the following:
- Download of circuits
- Storage of circuits and their associated metadata
- Updating circuits at regular intervals

Essentially, this is a centralized location where the competition can easily access circuits to test them.
"""

import bittensor as bt
import collections
from execution_layer.circuit import Circuit
import requests
import yarl
import os


class CircuitHandler:
    """
    Circuit handler is responsible for handling circuits submitted by miners for competitions.
    """

    def __init__(self, subtensor: bt.subtensor, metagraph: bt.metagraph):
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.circuits = collections.defaultdict(Circuit)
        self.cids = collections.defaultdict(str)

    def sync_circuits(self):
        for hotkey in self.metagraph.hotkeys:
            uid = self.metagraph.hotkeys.index(hotkey)
            cid = self.subtensor.get_commitment(self.metagraph.netuid, uid)
            if cid not in self.cids[hotkey] and cid is not None:
                self.cids[hotkey] = cid

    def download_circuit(self, hotkey: str) -> Circuit:
        cid = self.cids[hotkey]
        if cid is None:
            raise ValueError(f"No CID found for hotkey {hotkey}")

        ipfs_url = str(
            yarl.URL("https://ipfs.io/ipfs")
            / cid
            % {"download": "true", "format": "tar", "filename": "circuit.tar"}
        )
        response = requests.get(ipfs_url, stream=True)

        if response.status_code != 200:
            raise ValueError(f"Failed to download circuit for hotkey {hotkey}")

        tar_path = os.path.join("miner_circuits", f"{cid}.tar")
        with open(tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        circuit_dir = os.path.join("miner_circuits", cid)
        os.makedirs(circuit_dir, exist_ok=True)
        os.system(f"tar xf {tar_path} -C {circuit_dir}")

        os.remove(tar_path)

        return Circuit(cid)

    pass
