import traceback

import bittensor as bt

from _validator.validator_session import ValidatorSession
from constants import Roles
from utils import run_shared_preflight_checks

if __name__ == "__main__":
    run_shared_preflight_checks(Roles.VALIDATOR)

    try:
        # Initialize the circuit store and load external models
        from deployment_layer.circuit_store import circuit_store

        circuit_store.load_circuits()

        bt.logging.info("Creating validator session...")
        validator_session = ValidatorSession()
        bt.logging.info("Running main loop...")
        validator_session.run()
    except Exception as e:
        bt.logging.error("Critical error while attempting to run validator: ", e)
        traceback.print_exc()
