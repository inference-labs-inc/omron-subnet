from __future__ import annotations

import sys

import bittensor as bt

import config
from _validator.config import ValidatorConfig
from _validator.core.validator_loop import ValidatorLoop
from utils import clean_temp_files


class ValidatorSession:
    def __init__(self):
        self.config = ValidatorConfig(config.config)
        self.validator_loop = ValidatorLoop(self.config)

    def run(self):
        """
        Start the validator session and run the main loop
        """
        bt.logging.debug("Validator session started")

        try:
            self.validator_loop.run()
        except KeyboardInterrupt:
            bt.logging.info("KeyboardInterrupt caught. Exiting validator.")
            clean_temp_files()
            sys.exit(0)
