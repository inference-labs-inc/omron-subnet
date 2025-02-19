import asyncio
import os
import sys
import atexit

import pytest
from bittensor import Subtensor, Balance, Metagraph
from bittensor.core.extrinsics.set_weights import _do_set_weights
from bittensor.utils import networking

from tests.e2e_tests.tools.chain_interactions import register_subnet, wait_interval
from tests.e2e_tests.tools.e2e_test_utils import setup_wallet

FAST_BLOCKS_SPEEDUP_FACTOR = 5

# Track the miner process globally for cleanup
miner_process = None
# Track the validator process globally for cleanup
validator_process = None

def _build_command(wallet, netuid, process_type="miner"):
    """Helper to construct the command for starting a nuron process with the given wallet and netuid."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "..", "neurons", f"{process_type}.py")  # Relative path
    script_path = os.path.abspath(script_path)  # Convert to an absolute path for safety
    options = {
        "--netuid": str(netuid),
        "--subtensor.network": "local",
        "--subtensor.chain_endpoint": "ws://localhost:9945",
        "--wallet.path": wallet.path,
        "--wallet.name": wallet.name,
        "--wallet.hotkey": "default",
        # "--disable-wandb": None,  # TODO: why does this break the test?
    }
    return [sys.executable, script_path] + [str(arg) for pair in options.items() for arg in pair]


async def start_miner(wallet, netuid):
    """
    Starts the miner process and assigns it to a global variable for cleanup.

    Args:
        wallet: The wallet instance containing mining wallet configuration.
        netuid: The network UID for registering the miner.

    Returns:
        A subprocess process instance representing the running miner.
    """
    global miner_process
    command = _build_command(wallet, netuid, "miner")  # Construct the command dynamically

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ,  # Preserved environment variables
    )
    miner_process = process  # Global assignment remains for cleanup handling
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

async def start_validator(wallet, netuid):
    """
    Starts the validator process and assigns it to a global variable for cleanup.

    Args:
        wallet: The wallet instance containing mining wallet configuration.
        netuid: The network UID for registering the validator.

    Returns:
        A subprocess process instance representing the running validator.
    """
    global validator_process
    command = _build_command(wallet, netuid, "validator")  # Construct the command dynamically

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ,  # Preserved environment variables
    )
    validator_process = process  # Global assignment remains for cleanup handling
    return process

def cleanup_validator():
    """Ensures the validator is terminated properly even on test interruption."""
    global validator_process
    if validator_process and validator_process.returncode is None:  # Check if still running
        print("Cleaning up validator process...")
        validator_process.terminate()
        try:
            validator_process.wait(timeout=5)  # Give it time to exit
        except Exception:
            pass  # Ignore timeout errors
        print("validator process terminated.")

# Ensure cleanup on exit (even on CTRL+C)
atexit.register(cleanup_validator)
# Ensure cleanup on exit (even on CTRL+C)
atexit.register(cleanup_miner)

@pytest.mark.asyncio
async def test_emissions(local_chain):

    print("Testing emissions")
    netuid = 2
    # Register root as Alice - the subnet owner - automatically a neuron on network
    alice_keypair, alice_wallet = setup_wallet("//Alice")

    # # Register a subnet, netuid 1 --> dont need this as this automatically happens
    assert register_subnet(local_chain, alice_wallet), "Subnet wasn't created"

    # Verify subnet <netuid 1> created successfully
    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [netuid]
    ).serialize(), "Subnet wasn't created successfully"

    # Register Bob
    bob_keypair, bob_wallet = setup_wallet("//Bob")

    subtensor = Subtensor(network="ws://localhost:9945")

    # Register Bob to the network
    assert subtensor.burned_register(
        bob_wallet, netuid
    ), "Unable to register Bob as a neuron"

    # Assert two neurons are in network
    assert (
        len(subtensor.neurons(netuid=netuid)) >= 2
    ), "Alice & Bob not registered in the subnet"

    # Stake to become to top neuron after the first epoch
    assert subtensor.get_balance(bob_keypair.ss58_address) >= Balance.from_tao(100_000)
    subtensor.add_stake(bob_wallet, bob_keypair.ss58_address, netuid, Balance.from_tao(100_000), True, True)

    # miner output log
    log_file_path = os.path.join(os.path.dirname(__file__), "logs", "test_miner_emissions.log")

    # Open the log file in append mode
    log_file = open(log_file_path, "w")

    # Start the miner
    process = await start_miner(bob_wallet, netuid)

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

    # # wait for miner to settle in TODO: DO I need to wait?
    # await wait_interval(100, subtensor, netuid)

    # Validator log path
    log_file_path = os.path.join(os.path.dirname(__file__), "logs", "test_validator_emissions.log")

    # Open the log file in append mode
    log_file = open(log_file_path, "w")

    # Start the validator
    process = await start_validator(alice_wallet, netuid)

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

    print("validator process started in the background")

    # wait full epoch for validator to settle in
    await wait_interval(360, subtensor, netuid)

    # Set weights by Alice on the subnet
    _do_set_weights(
        subtensor=subtensor,
        wallet=alice_wallet,
        uids=[1],
        vals=[65535],
        netuid=netuid,
        version_key=0,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=5 * FAST_BLOCKS_SPEEDUP_FACTOR,
    )

    print("Alice neuron set weights successfully")

    await wait_interval(720, subtensor, netuid, 10)

    # Refresh metagraph
    metagraph = Metagraph(netuid=netuid, network="ws://localhost:9945")

    # Get current emissions and validate that Alice has gotten tao
    bob_neuron = metagraph.neurons[1]
    assert bob_neuron.incentive == 1
    assert bob_neuron.consensus == 1
    assert bob_neuron.rank == 1
    assert bob_neuron.trust == 1

    alice_neuron = metagraph.neurons[2]
    assert alice_neuron.validator_permit is True
    assert alice_neuron.dividends == 1
    assert alice_neuron.stake.tao == 10_000.0
    assert alice_neuron.validator_trust == 1


    print("✅ Passed test_incentive")
    cleanup_miner()
    cleanup_validator()