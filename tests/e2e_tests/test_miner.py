import asyncio
import os
import sys
import atexit

import bittensor
import pytest
from bittensor import Subtensor
from bittensor.utils import networking

from tests.e2e_tests.tools.chain_interactions import register_subnet
from tests.e2e_tests.tools.e2e_test_utils import setup_wallet




# Track the miner process globally for cleanup
miner_process = None

async def start_miner(wallet, netuid):
    """Starts the miner process and tracks it globally for cleanup."""
    global miner_process

    cmd = [
        sys.executable,
        "/Users/danielivanov/Repos/omron-subnet/neurons/miner.py",
        "--netuid", str(netuid),
        "--subtensor.network", "local",
        "--subtensor.chain_endpoint", "ws://localhost:9945",
        "--wallet.path", wallet.path,
        "--wallet.name", wallet.name,
        "--wallet.hotkey", "default"
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ,
    )

    miner_process = process
    return process

def cleanup_miner():
    """Ensures the miner is terminated properly even on test interruption."""
    global miner_process
    if miner_process and miner_process.returncode is None:  # Check if still running
        print("Cleaning up miner process...")
        miner_process.terminate()
        try:
            miner_process.wait(timeout=5)  # Give it time to exit
        except Exception:
            pass  # Ignore timeout errors
        print("Miner process terminated.")

# Ensure cleanup on exit (even on CTRL+C)
atexit.register(cleanup_miner)

@pytest.mark.asyncio
async def test_miner(local_chain):
    """
    Test the Axon mechanism and successful registration on the network.

    Steps:
        1. Register a subnet and register Alice
        2. Check if metagraph.axon is updated and check axon attributes
        3. Run Alice as a miner on the subnet
        4. Check the metagraph again after running the miner and verify all attributes
    Raises:
        AssertionError: If any of the checks or verifications fail
    """

    print("Testing test_axon")

    netuid = 1
    # Register root as Alice - the subnet owner
    alice_keypair, wallet = setup_wallet("//Alice")

    subtensor = Subtensor(network="ws://localhost:9945")

    # Register a subnet, netuid 1
    assert register_subnet(local_chain, wallet), "Subnet wasn't created"

    # Verify subnet <netuid 1> created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [netuid]
    ).serialize(), "Subnet wasn't created successfully"

    # Register Alice to the network
    assert subtensor.burned_register(
        wallet, netuid
    ), f"Neuron wasn't registered to subnet {netuid}"

    metagraph = subtensor.metagraph(netuid=netuid)

    # Validate current metagraph stats
    old_axon = metagraph.axons[1]
    # assert len(metagraph.axons) == 1, f"Expected 1 axon, but got {len(metagraph.axons)}"
    assert old_axon.hotkey == alice_keypair.ss58_address, "Hotkey mismatch for the axon"
    assert (
        old_axon.coldkey == alice_keypair.ss58_address
    ), "Coldkey mismatch for the axon"
    assert old_axon.ip == "0.0.0.0", f"Expected IP 0.0.0.0, but got {old_axon.ip}"
    assert old_axon.port == 0, f"Expected port 0, but got {old_axon.port}"
    assert old_axon.ip_type == 0, f"Expected IP type 0, but got {old_axon.ip_type}"

    log_file_path = "miner.log"

    # Open the log file in append mode
    log_file = open(log_file_path, "w")

    # Start the miner
    process = await start_miner(wallet, netuid)

    # Function to read and log output
    async def log_output(stream, log_prefix):
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded_line = line.decode().strip()
            print(f"{log_prefix}{decoded_line}")  # Print to console
            log_file.write(f"{log_prefix}{decoded_line}\n")  # Write to file
            log_file.flush()

    # Start logging both stdout and stderr in the background
    asyncio.create_task(log_output(process.stdout, "[STDOUT] "))
    asyncio.create_task(log_output(process.stderr, "[STDERR] "))

    print("Miner process started in the background")

    # Continue with other test steps
    await asyncio.sleep(15)

    print("Neuron Alice is now mining")

    # Refresh the metagraph
    metagraph = subtensor.metagraph(netuid=netuid)
    updated_axon = metagraph.axons[1]
    external_ip = networking.get_external_ip()

    # Assert updated attributes
    # assert (
    #     len(metagraph.axons) == 1
    # ), f"Expected 1 axon, but got {len(metagraph.axons)} after mining"

    # assert (
    #     len(metagraph.neurons) == 1
    # ), f"Expected 1 neuron, but got {len(metagraph.neurons)}"

    print(f"Metagraph updated axon IP: {updated_axon.ip}, Expected: {external_ip}")
    assert (
            updated_axon.ip == external_ip
    ), f"Expected IP {external_ip}, but got {updated_axon.ip}"

    assert (
        updated_axon.ip_type == networking.ip_version(external_ip)
    ), f"Expected IP type {networking.ip_version(external_ip)}, but got {updated_axon.ip_type}"

    assert updated_axon.port == 8091, f"Expected port 8091, but got {updated_axon.port}"

    assert (
        updated_axon.hotkey == alice_keypair.ss58_address
    ), "Hotkey mismatch after mining"

    assert (
        updated_axon.coldkey == alice_keypair.ss58_address
    ), "Coldkey mismatch after mining"

    print("✅ Passed test_axon")

    # Ensure the miner process is terminated
    cleanup_miner()
