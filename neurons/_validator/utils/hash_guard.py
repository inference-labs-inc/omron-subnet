from execution_layer.base_input import BaseInput
import bittensor as bt


class HashGuard:
    """
    A safety checker to ensure input data is never repeated.
    """

    MAX_HASHES = 8192

    def __init__(self):
        self.hashes = []

    def check_hash(self, input: BaseInput) -> None:
        hash_value = hash(frozenset(input.data.items()))
        if hash_value in self.hashes:
            bt.logging.error(f"Hash already exists: {hash_value}. Inputs: {input.data}")
            raise ValueError("Hash already exists")

        self.hashes.append(hash_value)
        if len(self.hashes) > self.MAX_HASHES:
            self.hashes.pop(0)