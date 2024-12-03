from execution_layer.base_input import BaseInput
import bittensor as bt
import json


class HashGuard:
    """
    A safety checker to ensure input data is never repeated.
    """

    MAX_HASHES = 8192

    def __init__(self):
        self.hashes = []

    def check_hash(self, input: BaseInput) -> None:
        hash_value = hash(json.dumps(input))
        if hash_value in self.hashes:
            bt.logging.error(f"Hash already exists: {hash_value}. Inputs: {input}")
            raise ValueError("Hash already exists")

        self.hashes.append(hash_value)
        if len(self.hashes) > self.MAX_HASHES:
            self.hashes.pop(0)
