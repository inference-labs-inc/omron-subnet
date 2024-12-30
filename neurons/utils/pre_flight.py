import asyncio
import json
import os
import subprocess
import time
import traceback
from functools import partial
from typing import Optional

# trunk-ignore(pylint/E0611)
import bittensor as bt
import ezkl
import requests

import cli_parser
from constants import IGNORED_MODEL_HASHES
from execution_layer.circuit import ProofSystem

LOCAL_SNARKJS_INSTALL_DIR = os.path.join(os.path.expanduser("~"), ".snarkjs")
LOCAL_SNARKJS_PATH = os.path.join(
    LOCAL_SNARKJS_INSTALL_DIR, "node_modules", ".bin", "snarkjs"
)
LOCAL_EZKL_PATH = os.path.join(os.path.expanduser("~"), ".ezkl", "ezkl")
TOOLCHAIN = "nightly-2024-09-30"
JOLT_VERSION = "dd9e5c4bcf36ffeb75a576351807f8d86c33ec66"


async def download_srs(logrows):
    await ezkl.get_srs(logrows=logrows, commitment=ezkl.PyCommitments.KZG)


def run_shared_preflight_checks(role: Optional[str] = None):
    """
    This function executes a series of checks to ensure the environment is properly
    set up for both validator and miner operations.
    Checks:
    - Model files are synced up
    - Node.js >= 20 is installed
    - SnarkJS is installed
    - Rust and Cargo are installed
    - Rust nightly toolchain is installed
    - Jolt is installed

    Raises:
        Exception: If any of the pre-flight checks fail.
    """

    preflight_checks = [
        ("Init configs", partial(cli_parser.init_config, role=role)),
        ("Syncing model files", sync_model_files),
        ("Ensuring Node.js version", ensure_nodejs_version),
        ("Checking SnarkJS installation", ensure_snarkjs_installed),
        ("Checking EZKL installation", ensure_ezkl_installed),
    ]

    bt.logging.info(" PreFlight | Running pre-flight checks")

    # Skip sync_model_files during docker build
    if os.getenv("OMRON_DOCKER_BUILD", False):
        bt.logging.info(" PreFlight | Skipping model file sync")
        preflight_checks.remove(("Syncing model files", sync_model_files))

    for check_name, check_function in preflight_checks:
        bt.logging.info(f" PreFlight | {check_name}")
        try:
            check_function()
            bt.logging.success(f" PreFlight | {check_name} completed successfully")
        except Exception as e:
            bt.logging.error(f"Failed {check_name.lower()}.", e)
            bt.logging.debug(f" PreFlight | {check_name} error details: {str(e)}")
            traceback.print_exc()
            raise e

    bt.logging.info(" PreFlight | Pre-flight checks completed.")


def ensure_ezkl_installed():
    """
    Ensure EZKL is installed by first checking if it exists, and if not,
    running the official installation script. Also verifies the version matches.
    """
    python_ezkl_version = ezkl.__version__
    os.environ["EZKL_REPO_PATH"] = os.path.join(
        os.path.dirname(cli_parser.config.full_path), "ezkl"
    )
    try:
        if os.path.exists(LOCAL_EZKL_PATH):
            # Check version matches
            result = subprocess.run(
                [LOCAL_EZKL_PATH, "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            if python_ezkl_version in result.stdout:
                bt.logging.info(
                    f"EZKL is already installed with correct version: {python_ezkl_version}"
                )
                return
            else:
                bt.logging.warning("EZKL version mismatch, reinstalling...")

        # trunk-ignore(bandit/B605)
        subprocess.run(
            f"curl -s https://raw.githubusercontent.com/zkonduit/ezkl/main/install_ezkl_cli.sh | bash -s -- v{python_ezkl_version}",  # noqa
            shell=True,
            check=True,
        )
        bt.logging.info("EZKL installed successfully")

    except subprocess.CalledProcessError as e:
        bt.logging.error(f"Failed to install/verify EZKL: {e}")
        raise RuntimeError(
            "EZKL installation failed. Please install it manually."
        ) from e


def ensure_snarkjs_installed():
    """
    Ensure snarkjs is installed and available for use in a local .snarkjs directory.
    """

    try:
        # trunk-ignore(bandit/B603)
        subprocess.run(
            [LOCAL_SNARKJS_PATH, "r1cs", "info", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        bt.logging.info(
            "snarkjs is already installed and available in the local directory."
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        bt.logging.warning(
            "snarkjs not found in local directory. Attempting to install..."
        )
        try:
            # Create the local installation directory if it doesn't exist
            os.makedirs(LOCAL_SNARKJS_INSTALL_DIR, exist_ok=True)

            # Install snarkjs in the local directory
            # trunk-ignore(bandit/B603)
            # trunk-ignore(bandit/B607)
            subprocess.run(
                [
                    "npm",
                    "install",
                    "--prefix",
                    LOCAL_SNARKJS_INSTALL_DIR,
                    "snarkjs@0.7.4",
                ],
                check=True,
            )
            bt.logging.info(
                "snarkjs has been successfully installed in the local directory."
            )
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"Failed to install snarkjs: {e}")
            raise RuntimeError(
                "snarkjs installation failed. Please install it manually."
            ) from e


def sync_model_files():
    """
    Sync external model files
    """
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "deployment_layer")
    SYNC_LOG_PREFIX = "  SYNC  | "

    for model_hash in os.listdir(MODEL_DIR):
        if not model_hash.startswith("model_"):
            continue

        if model_hash.split("_")[1] in IGNORED_MODEL_HASHES:
            bt.logging.info(
                SYNC_LOG_PREFIX
                + f"Ignoring model {model_hash} as it is in the ignored list."
            )
            continue

        metadata_file = os.path.join(MODEL_DIR, model_hash, "metadata.json")
        if not os.path.isfile(metadata_file):
            bt.logging.error(
                SYNC_LOG_PREFIX
                + f"Metadata file not found at {metadata_file} for {model_hash}. Skipping sync for this model."
            )
            continue

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            bt.logging.error(
                SYNC_LOG_PREFIX + f"Failed to parse JSON from {metadata_file}"
            )
            continue
        # If it's an EZKL model, we'll try to download the SRS files
        if metadata.get("proof_system") == ProofSystem.EZKL:
            ezkl_settings_file = os.path.join(MODEL_DIR, model_hash, "settings.json")
            if not os.path.isfile(ezkl_settings_file):
                bt.logging.error(
                    f"{SYNC_LOG_PREFIX}Settings file not found at {ezkl_settings_file} for {model_hash}. Skipping sync."
                )
                continue

            try:
                with open(ezkl_settings_file, "r", encoding="utf-8") as f:
                    logrows = json.load(f).get("run_args", {}).get("logrows")
                    if logrows:
                        loop = asyncio.get_event_loop()
                        loop.run_until_complete(download_srs(logrows))
                        bt.logging.info(
                            f"{SYNC_LOG_PREFIX}Successfully downloaded SRS for logrows={logrows}"
                        )
            except (json.JSONDecodeError, subprocess.CalledProcessError) as e:
                bt.logging.error(
                    f"{SYNC_LOG_PREFIX}Failed to process settings or download SRS: {e}"
                )
                continue

        external_files = metadata.get("external_files", {})
        for key, url in external_files.items():
            file_path = os.path.join(
                cli_parser.config.external_model_dir, model_hash, key
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if os.path.isfile(file_path):
                bt.logging.info(
                    SYNC_LOG_PREFIX
                    + f"File {key} for {model_hash} already downloaded, skipping..."
                )
                continue

            bt.logging.info(SYNC_LOG_PREFIX + f"Downloading {url} to {file_path}...")
            try:
                with requests.get(url, timeout=600, stream=True) as response:
                    response.raise_for_status()
                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
            except requests.RequestException as e:
                bt.logging.error(
                    SYNC_LOG_PREFIX + f"Failed to download {url} to {file_path}: {e}"
                )
                continue


def ensure_nodejs_version():
    """
    Ensure that Node.js version 20 is installed
    If not installed, provide instructions for manual installation.
    """
    NODE_LOG_PREFIX = "  NODE  | "

    try:
        node_version = subprocess.check_output(["node", "--version"]).decode().strip()
        npm_version = subprocess.check_output(["npm", "--version"]).decode().strip()

        if node_version.startswith("v20."):
            bt.logging.info(
                NODE_LOG_PREFIX
                + f"Node.js version {node_version} and npm version {npm_version} are installed."
            )
            return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    bt.logging.error(
        NODE_LOG_PREFIX + "Node.js is not installed or is not the correct version."
    )
    bt.logging.error(
        NODE_LOG_PREFIX
        + "\033[91mPlease install Node.js >= 20 using the following command\n./setup.sh --no-install\033[0m"
    )

    time.sleep(10)
    raise RuntimeError(
        "Node.js >= 20 is required but not installed. Please install it manually and restart the process."
    )


def ensure_rust_cargo_installed():
    """
    Ensure that Rust and Cargo are installed.
    If not installed, install them and instruct the user to restart the shell and PM2 process.
    """
    RUST_LOG_PREFIX = "  RUST  | "

    try:
        subprocess.run(
            [
                os.path.join(os.path.expanduser("~"), ".cargo", "bin", "rustc"),
                "--version",
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                os.path.join(os.path.expanduser("~"), ".cargo", "bin", "cargo"),
                "--version",
            ],
            check=True,
            capture_output=True,
        )
        bt.logging.info(f"{RUST_LOG_PREFIX}Rust and Cargo are already installed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        bt.logging.info(f"{RUST_LOG_PREFIX}Rust and/or Cargo not found. Installing...")
        try:
            rustup_script = requests.get("https://sh.rustup.rs").text
            subprocess.run(
                ["sh", "-s", "--", "-y"],
                input=rustup_script.encode(),
                check=True,
                shell=False,
            )
            cargo_path = os.path.join(os.path.expanduser("~"), ".cargo", "bin", "cargo")
            if not os.path.exists(cargo_path):
                bt.logging.info(
                    f"{RUST_LOG_PREFIX}Cargo not found. Adding cargo component..."
                )
                try:
                    subprocess.run(
                        [
                            os.path.join(
                                os.path.expanduser("~"), ".cargo", "bin", "rustup"
                            ),
                            "component",
                            "add",
                            "cargo",
                        ],
                        check=True,
                        capture_output=True,
                    )
                    bt.logging.info(
                        f"{RUST_LOG_PREFIX}Cargo component added successfully."
                    )
                except subprocess.CalledProcessError as e:
                    bt.logging.error(
                        f"{RUST_LOG_PREFIX}Failed to add cargo component: {e}"
                    )
                    raise RuntimeError("Failed to add cargo component.") from e
            bt.logging.info(
                f"{RUST_LOG_PREFIX}Rust and Cargo have been successfully installed."
            )

            bt.logging.info(f"{RUST_LOG_PREFIX}Installation complete.")
            bt.logging.info(
                f"{RUST_LOG_PREFIX}\033[93mIMPORTANT: Ensure you have pkg-config, libssl-dev and openssl installed"
                "with sudo apt install pkg-config libssl-dev openssl.\033[0m"
            )
            bt.logging.info(
                f"{RUST_LOG_PREFIX}\033[93mPausing. To complete install, restart your machine using sudo reboot.\033[0m"
            )
            time.sleep(1e9)

        except subprocess.CalledProcessError as e:
            bt.logging.error(f"{RUST_LOG_PREFIX}Failed to install Rust and Cargo: {e}")
            raise RuntimeError(
                "Rust and Cargo installation failed. Please install them manually."
            ) from e


def ensure_rust_nightly_installed():
    """
    Ensure that the Rust nightly toolchain is installed with the specified target.
    If not installed, install it.
    """
    RUST_LOG_PREFIX = "  RUST  | "

    try:
        result = subprocess.run(
            [f"{os.path.expanduser('~')}/.cargo/bin/rustup", "toolchain", "list"],
            check=True,
            capture_output=True,
            text=True,
        )
        if TOOLCHAIN in result.stdout:
            result = subprocess.run(
                [
                    f"{os.path.expanduser('~')}/.cargo/bin/rustup",
                    "target",
                    "list",
                    "--installed",
                    "--toolchain",
                    TOOLCHAIN,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
    except subprocess.CalledProcessError:
        pass

    bt.logging.info(f"{RUST_LOG_PREFIX}Installing Rust {TOOLCHAIN}...")
    try:
        subprocess.run(
            [
                f"{os.path.expanduser('~')}/.cargo/bin/rustup",
                "toolchain",
                "install",
                TOOLCHAIN,
            ],
            check=True,
        )
        bt.logging.info(
            f"{RUST_LOG_PREFIX}Rust {TOOLCHAIN} has been successfully installed."
        )
    except subprocess.CalledProcessError as e:
        bt.logging.error(f"{RUST_LOG_PREFIX}Failed to install Rust toolchain: {e}")
        raise RuntimeError(
            f"Rust {TOOLCHAIN} installation failed. Please install it manually."
        ) from e


def ensure_jolt_installed():
    """
    Ensure that Jolt is installed.
    If not installed, install it and the toolchain.
    """
    JOLT_LOG_PREFIX = "  JOLT  | "

    try:
        subprocess.run(
            [
                os.path.join(os.path.expanduser("~"), ".cargo", "bin", "jolt"),
                "--version",
            ],
            check=True,
            capture_output=True,
        )
        bt.logging.info(f"{JOLT_LOG_PREFIX}Jolt is already installed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        bt.logging.info(f"{JOLT_LOG_PREFIX}Jolt not found. Installing...")
        try:
            subprocess.run(
                [
                    os.path.join(os.path.expanduser("~"), ".cargo", "bin", "cargo"),
                    f"+{TOOLCHAIN}",
                    "install",
                    "--git",
                    "https://github.com/a16z/jolt",
                    "--rev",
                    JOLT_VERSION,
                    "--force",
                    "--bins",
                    "jolt",
                ],
                check=True,
            )
            bt.logging.info(f"{JOLT_LOG_PREFIX}Jolt has been successfully installed.")
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"{JOLT_LOG_PREFIX}Failed to install Jolt: {e}")
            raise RuntimeError(
                "Jolt installation failed. Please install it manually."
            ) from e

    bt.logging.info(f"{JOLT_LOG_PREFIX}Running jolt install-toolchain...")
    try:
        subprocess.run(
            [
                os.path.join(os.path.expanduser("~"), ".cargo", "bin", "jolt"),
                "install-toolchain",
            ],
            check=True,
        )
        bt.logging.info(
            f"{JOLT_LOG_PREFIX}jolt install-toolchain completed successfully."
        )
    except subprocess.CalledProcessError as e:
        bt.logging.error(f"{JOLT_LOG_PREFIX}Failed to run jolt install-toolchain: {e}")
        raise RuntimeError(
            "jolt install-toolchain failed. Please run it manually."
        ) from e


def compile_jolt_circuits():
    """
    Compile Jolt circuits for each model that uses the Jolt proof system
    """
    JOLT_LOG_PREFIX = "  JOLT  | "
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "deployment_layer")

    for model_hash in os.listdir(MODEL_DIR):
        if not model_hash.startswith("model_"):
            continue

        if model_hash.split("_")[1] in IGNORED_MODEL_HASHES:
            bt.logging.info(
                JOLT_LOG_PREFIX
                + f"Ignoring model {model_hash} as it is in the ignored list."
            )
            continue

        metadata_file = os.path.join(MODEL_DIR, model_hash, "metadata.json")
        if not os.path.isfile(metadata_file):
            bt.logging.warning(
                f"{JOLT_LOG_PREFIX}Metadata file not found for {model_hash}. Skipping."
            )
            continue

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            bt.logging.error(
                f"{JOLT_LOG_PREFIX}Failed to parse JSON from {metadata_file}"
            )
            continue

        if metadata.get("proof_system") != "JOLT":
            bt.logging.info(f"{JOLT_LOG_PREFIX}Skipping non-Jolt circuit: {model_hash}")
            continue

        circuit_path = os.path.join(
            MODEL_DIR, model_hash, "target", "release", "circuit"
        )
        if os.path.exists(circuit_path):
            bt.logging.info(
                f"{JOLT_LOG_PREFIX}Circuit already compiled for {model_hash}"
            )
            continue

        bt.logging.info(f"{JOLT_LOG_PREFIX}Compiling circuit for {model_hash}")
        try:
            subprocess.run(
                ["cargo", "build", "--release"],
                cwd=os.path.join(MODEL_DIR, model_hash),
                check=True,
            )
            bt.logging.info(
                f"{JOLT_LOG_PREFIX}Successfully compiled circuit for {model_hash}"
            )
        except subprocess.CalledProcessError as e:
            bt.logging.error(
                f"{JOLT_LOG_PREFIX}Failed to compile circuit for {model_hash}: {e}"
            )

    bt.logging.info(f"{JOLT_LOG_PREFIX}Jolt circuit compilation process completed.")


def is_safe_path(base_path, path):
    return os.path.realpath(path).startswith(os.path.realpath(base_path))


def safe_extract(tar, path):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_safe_path(path, member_path):
            continue
        tar.extract(member, path)
