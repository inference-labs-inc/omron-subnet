import os
import re
import shlex
import signal
import subprocess
import time
import threading

import pytest
from substrateinterface import SubstrateInterface

from bittensor.utils.btlogging import logging


# Fixture for setting up and tearing down a localnet.sh chain between tests
@pytest.fixture(scope="function")
def local_chain(request):
    param = request.param if hasattr(request, "param") else None
    # Get the environment variable for the script path
    script_path = os.getenv("LOCALNET_SH_PATH")

    if not script_path:
        # Skip the test if the localhost.sh path is not set
        logging.warning("LOCALNET_SH_PATH env variable is not set, e2e test skipped.")
        pytest.skip("LOCALNET_SH_PATH environment variable is not set.")

    # Check if param is None, and handle it accordingly
    args = "" if param is None else f"{param}"

    # Compile commands to send to process
    cmds = shlex.split(f"{script_path} {args}")

    # Start new node process
    process = subprocess.Popen(
        cmds,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )

    # Pattern match indicates node is compiled and ready
    pattern = re.compile(r"Imported #1")

    timestamp = int(time.time())

    def wait_for_node_start(process, pattern):
        while True:
            line = process.stdout.readline()
            if not line:
                break

            print(line.strip())
            # 10 min as timeout
            if int(time.time()) - timestamp > 10 * 60:
                print("Subtensor not started in time")
                return
            if pattern.search(line):
                print("Node started!")
                break

        # Start a background reader after pattern is found
        # To prevent the buffer filling up
        def read_output():
            while True:
                line = process.stdout.readline()
                if not line:
                    break

        reader_thread = threading.Thread(target=read_output, daemon=True)
        reader_thread.start()

    wait_for_node_start(process, pattern)

    # Run the test, passing in substrate interface
    yield SubstrateInterface(url="ws://127.0.0.1:9945")

    # Terminate the process group (includes all child processes)
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    # Give some time for the process to terminate
    time.sleep(1)

    # If the process is not terminated, send SIGKILL
    if process.poll() is None:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)

    # Ensure the process has terminated
    process.wait()
