"""
Entry point for running just preflight checks:
 - Check CLI args (in fact completely unnecessary here)
 - Model files are synced up
 - Node.js >= 20 is installed
 - SnarkJS is installed
 - Rust and Cargo are installed
 - Rust nightly toolchain is installed
 - Jolt is installed

This script is created to be called during the Docker image build process
to ensure that all dependencies are installed.
"""

import bittensor as bt

from utils import run_shared_preflight_checks

if __name__ == "__main__":
    # Run preflight checks, and that's it
    run_shared_preflight_checks()

    bt.logging.info("Preflight checks completed. Exiting...")
