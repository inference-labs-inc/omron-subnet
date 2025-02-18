import asyncio
import os
import sys
import atexit

import pytest
from bittensor import Subtensor, Metagraph, Balance
from bittensor.utils import networking

from tests.e2e_tests.tools.chain_interactions import register_subnet, wait_interval
from tests.e2e_tests.tools.e2e_test_utils import setup_wallet

# Track the validator process globally for cleanup
validator_process = None


def _build_command(wallet, netuid):
    """Helper to construct the command for starting the validator."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "..", "neurons", "validator.py")  # Relative path
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
    command = _build_command(wallet, netuid)  # Construct the command dynamically

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

@pytest.mark.asyncio
async def test_validator(local_chain):
    """
    Test the Dendrite mechanism and successful registration on the network.

    Steps:
        1. Register a subnet and register Alice
        2. Check if metagraph.dendrite is updated and check dendrite attributes
        3. Run Alice as a validator on the subnet
        4. Check the metagraph again after running the validator and verify all attributes
    Raises:
        AssertionError: If any of the checks or verifications fail
    """

    print("Testing Validator")

    netuid = 2
    # Register root as Alice - the subnet owner - automatically a neuron on network
    alice_keypair, wallet = setup_wallet("//Alice")

    # # Register a subnet, netuid 1 --> dont need this as this automatically happens
    assert register_subnet(local_chain, wallet), "Subnet wasn't created"

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

    metagraph = subtensor.metagraph(netuid=netuid)

    # Assert one extra neuron is Bob
    assert len(subtensor.neurons(netuid=netuid)) == 2 # Alice registered the network, and bob is validator
    neuron = metagraph.neurons[1]
    assert neuron.hotkey == bob_keypair.ss58_address
    assert neuron.coldkey == bob_keypair.ss58_address

    # Assert stake is 0
    assert neuron.stake.tao == 0

    # Stake to become to top neuron after the first epoch
    assert subtensor.get_balance(bob_keypair.ss58_address) >= Balance.from_tao(100_000)
    subtensor.add_stake(bob_wallet, bob_keypair.ss58_address, netuid, Balance.from_tao(100_000), True, True)

    await wait_interval(100, subtensor, netuid)

    # Refresh metagraph
    metagraph = Metagraph(netuid=netuid, network="ws://localhost:9945")
    old_neuron = metagraph.neurons[1]

    # Assert stake in alpha is > 0
    assert (
        old_neuron.stake.tao > Balance(0)
    ), f"Expected greater than 0 staked TAO, but got {neuron.stake.tao}"

    # Assert neuron is not a validator yet
    assert old_neuron.active is True
    # assert old_neuron.validator_permit is False
    assert old_neuron.validator_trust == 0.0
    assert old_neuron.pruning_score == 65535

    log_file_path = os.path.join(os.path.dirname(__file__), "logs", "test_validator.log")

    # Open the log file in append mode
    log_file = open(log_file_path, "w")

    # Start the validator
    process = await start_validator(wallet, netuid)

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

    # Refresh the metagraph
    metagraph = Metagraph(netuid=netuid, network="ws://localhost:9945")

    # Refresh validator neuron
    updated_neuron = metagraph.neurons[1]

    # assert len(metagraph.neurons) == 1
    assert updated_neuron.active is True
    assert updated_neuron.validator_permit is True
    assert updated_neuron.hotkey == bob_keypair.ss58_address
    assert updated_neuron.coldkey == bob_keypair.ss58_address
    assert updated_neuron.pruning_score != 0

    print("✅ Passed test_validator")

    # Ensure the validator process is terminated
    cleanup_validator()
