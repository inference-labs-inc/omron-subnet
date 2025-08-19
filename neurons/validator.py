import traceback
import os

# isort: off
import cli_parser  # <- this need to stay before bittensor import

import bittensor as bt

# isort: on

from _validator.validator_session import ValidatorSession
from constants import Roles
from utils import run_shared_preflight_checks

if __name__ == "__main__":
    cli_parser.init_config(Roles.VALIDATOR)
    run_shared_preflight_checks(Roles.VALIDATOR)

    # Configure VizTracer for comprehensive flow tracing
    enable_tracing = os.getenv("ENABLE_VIZTRACER", "false").lower() == "true"

    if enable_tracing:
        from viztracer import VizTracer

        tracer = VizTracer(
            output_file=f"validator_trace_{os.getpid()}.json",
            log_func_entry=True,
            log_func_args=True,
            log_func_retval=True,
            log_exception=True,
            log_gc=True,
            log_async=True,
            log_multiprocess=True,
            max_stack_depth=50,
            min_duration=0,
            include_files=["neurons/", "_validator/", "utils/"],
            exclude_files=["venv/", "site-packages/"],
            log_torch=False,
            verbose=1,
        )

        tracer.start()
        bt.logging.info("VizTracer enabled - tracing validator execution flow")

    try:
        # Initialize the circuit store and load external models
        from deployment_layer.circuit_store import circuit_store

        circuit_store.load_circuits()

        bt.logging.info("Creating validator session...")
        validator_session = ValidatorSession()
        bt.logging.debug("Running main loop...")
        validator_session.run()
    except Exception as e:
        bt.logging.error("Critical error while attempting to run validator: ", e)
        traceback.print_exc()
    finally:
        if enable_tracing and "tracer" in locals():
            tracer.stop()
            tracer.save()
            bt.logging.info(f"Trace saved to validator_trace_{os.getpid()}.json")
