import traceback

import bittensor as bt
import config

from _validator.validator_session import ValidatorSession
from utils import run_shared_preflight_checks

if __name__ == "__main__":
    run_shared_preflight_checks(is_validator=True)

    try:
        # Initialize the circuit store and load external models
        from deployment_layer.circuit_store import circuit_store

        circuit_store.load_circuits(config.config.external_model_dir)

        bt.logging.info("Creating validator session...")
        validator_session = ValidatorSession(config.config)
        bt.logging.info("Running main loop...")
        validator_session.run()
    except Exception as e:
        bt.logging.error("Critical error while attempting to run validator: ", e)
        traceback.print_exc()
