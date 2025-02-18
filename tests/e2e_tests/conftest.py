import os
import re
import shlex
import signal
import socket
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

    # Check if there is already a process listening on the port (127.0.0.1:9945)
    def is_port_in_use(host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((host, port))
                return True
            except (ConnectionRefusedError, OSError):
                return False

    port = 9945
    if is_port_in_use('127.0.0.1', port):
        print(f"A process is already running on port {port}. Skipping start.")
        yield SubstrateInterface(url=f"ws://127.0.0.1:{port}")
        return

    # Check if param is None, and handle it accordingly
    args = "" if param is None else f"{param}"

    # Compile commands to send to process
    cmds = shlex.split(f"{script_path} {args}")

    # Create the logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Open the log file for writing
    log_file_path = os.path.join(logs_dir, "local_chain.log")
    log_file = open(log_file_path, "w", buffering=1)  # Line-buffered mode

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
            log_file.write(line)
            log_file.flush()

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
                # Log both to console and log file
                # print(line.strip())
                log_file.write(line)
                log_file.flush()

        reader_thread = threading.Thread(target=read_output, daemon=True)
        reader_thread.start()

    wait_for_node_start(process, pattern)

    try:
        # Pass the SubstrateInterface to the test
        yield SubstrateInterface(url="ws://127.0.0.1:9945")
    finally:
        # Terminate the process
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        time.sleep(1)
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()
        log_file.close()  # Close the log file
