import datetime
from _validator.config import ValidatorConfig


class ValidatorKeysCache:
    """
    A class to cache validator keys. This is used to reduce the number of requests to the metagraph.
    """

    def __init__(self, config: ValidatorConfig) -> None:
        self.cached_keys: dict[int, list[str]] = {}
        self.cached_timestamps: dict[int, datetime.datetime] = {}
        self.config: ValidatorConfig = config

    def fetch_validator_keys(self, netuid: int) -> None:
        """
        Fetch the validator keys for a given netuid and cache them.
        """
        self.cached_keys[netuid] = [
            neuron.hotkey
            for neuron in self.config.subtensor.neurons_lite(netuid)
            if neuron.validator_permit
        ]
        self.cached_timestamps[netuid] = datetime.datetime.now() + datetime.timedelta(
            hours=12
        )

    def check_validator_key(self, ss58_address: str, netuid: int) -> bool:
        """
        Check if a given key is a validator key for a given netuid.
        """
        if ss58_address in self.config.api.whitelisted_public_keys:
            # If the sender is whitelisted, we don't need to check the key
            return True
        cache_timestamp = self.cached_timestamps.get(netuid, None)
        if cache_timestamp is None or cache_timestamp > datetime.datetime.now():
            self.fetch_validator_keys(netuid)
        return ss58_address in self.cached_keys.get(netuid, [])
