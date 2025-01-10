from __future__ import annotations

import bittensor as bt
from _validator.config import ValidatorConfig
from _validator.core.validator_loop import ValidatorLoop
from utils import clean_temp_files
import sys
import asyncio


class ValidatorSession:
    def __init__(self, config: bt.config):
        self.config = ValidatorConfig(config)
        self.validator_loop = ValidatorLoop(self.config)

    def run(self):
        """
        Start the validator session and run the main loop
        """
        bt.logging.debug("Validator session started")

        try:
            asyncio.run(self.validator_loop.run())
        except KeyboardInterrupt:
            bt.logging.info("KeyboardInterrupt caught. Exiting validator.")
            clean_temp_files()
            sys.exit(0)
