import traceback
import os
import subprocess
import time
import json
import requests
import ezkl
import asyncio

# trunk-ignore(pylint/E0611)
from bittensor import logging

from constants import IGNORED_MODEL_HASHES
from execution_layer.circuit import ProofSystem


LOCAL_SNARKJS_INSTALL_DIR = os.path.join(os.path.expanduser("~"), ".snarkjs")
LOCAL_SNARKJS_PATH = os.path.join(
    LOCAL_SNARKJS_INSTALL_DIR, "node_modules", ".bin", "snarkjs"
)
TOOLCHAIN = "nightly-2024-09-30"
JOLT_VERSION = "9f0b9e6d95814dfe15d74ea736b9f89d505e8d07"


def run_shared_preflight_checks():
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
        ("Syncing model files", sync_model_files),
        ("Ensuring Node.js version", ensure_nodejs_version),
        ("Checking SnarkJS installation", ensure_snarkjs_installed),
        ("Checking Rust and Cargo installation", ensure_rust_cargo_installed),
        ("Checking Rust nightly toolchain", ensure_rust_nightly_installed),
        ("Checking Jolt installation", ensure_jolt_installed),
        ("Compiling Jolt circuits", compile_jolt_circuits),
    ]

    logging.info(" PreFlight | Running pre-flight checks")

    for check_name, check_function in preflight_checks:
        logging.info(f" PreFlight | {check_name}")
        try:
            check_function()
            logging.success(f" PreFlight | {check_name} completed successfully")
        except Exception as e:
            logging.error(f"Failed {check_name.lower()}.", e)
            logging.debug(f" PreFlight | {check_name} error details: {str(e)}")
            traceback.print_exc()
            raise e

    logging.info(" PreFlight | Pre-flight checks completed.")


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
        logging.info(
            "snarkjs is already installed and available in the local directory."
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning(
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
            logging.info(
                "snarkjs has been successfully installed in the local directory."
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install snarkjs: {e}")
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
            logging.info(
                SYNC_LOG_PREFIX
                + f"Ignoring model {model_hash} as it is in the ignored list."
            )
            continue

        metadata_file = os.path.join(MODEL_DIR, model_hash, "metadata.json")
        if not os.path.isfile(metadata_file):
            logging.error(
                SYNC_LOG_PREFIX
                + f"Metadata file not found at {metadata_file} for {model_hash}. Skipping sync for this model."
            )
            continue

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            logging.error(
                SYNC_LOG_PREFIX + f"Failed to parse JSON from {metadata_file}"
            )
            continue
        # If it's an EZKL model, we'll try to download the SRS files
        if metadata.get("proof_system") == ProofSystem.EZKL:
            ezkl_settings_file = os.path.join(MODEL_DIR, model_hash, "settings.json")
            if not os.path.isfile(ezkl_settings_file):
                logging.error(
                    f"{SYNC_LOG_PREFIX}Settings file not found at {ezkl_settings_file} for {model_hash}. Skipping sync."
                )
                continue

            try:
                with open(ezkl_settings_file, "r", encoding="utf-8") as f:
                    logrows = json.load(f).get("run_args", {}).get("logrows")
                    if logrows:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            ezkl.get_srs(
                                logrows=logrows, commitment=ezkl.PyCommitments.KZG
                            )
                        )
                        loop.close()
                        logging.info(
                            f"{SYNC_LOG_PREFIX}Successfully downloaded SRS for logrows={logrows}"
                        )
            except (json.JSONDecodeError, subprocess.CalledProcessError) as e:
                logging.error(
                    f"{SYNC_LOG_PREFIX}Failed to process settings or download SRS: {e}"
                )
                continue

        external_files = metadata.get("external_files", {})
        for key, url in external_files.items():
            file_path = os.path.join(MODEL_DIR, model_hash, key)
            if os.path.isfile(file_path):
                logging.info(
                    SYNC_LOG_PREFIX
                    + f"File {key} for {model_hash} already downloaded, skipping..."
                )
                continue

            logging.info(SYNC_LOG_PREFIX + f"Downloading {url} to {file_path}...")
            try:
                response = requests.get(url, timeout=600)
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    f.write(response.content)
            except requests.RequestException as e:
                logging.error(
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
            logging.info(
                NODE_LOG_PREFIX
                + f"Node.js version {node_version} and npm version {npm_version} are installed."
            )
            return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    logging.error(
        NODE_LOG_PREFIX + "Node.js is not installed or is not the correct version."
    )
    logging.error(
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
        logging.info(f"{RUST_LOG_PREFIX}Rust and Cargo are already installed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.info(f"{RUST_LOG_PREFIX}Rust and/or Cargo not found. Installing...")
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
                logging.info(
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
                    logging.info(
                        f"{RUST_LOG_PREFIX}Cargo component added successfully."
                    )
                except subprocess.CalledProcessError as e:
                    logging.error(
                        f"{RUST_LOG_PREFIX}Failed to add cargo component: {e}"
                    )
                    raise RuntimeError("Failed to add cargo component.") from e
            logging.info(
                f"{RUST_LOG_PREFIX}Rust and Cargo have been successfully installed."
            )

            logging.info(f"{RUST_LOG_PREFIX}Installation complete.")
            logging.info(
                f"{RUST_LOG_PREFIX}\033[93mIMPORTANT: Ensure you have pkg-config, libssl-dev and openssl installed"
                "with sudo apt install pkg-config libssl-dev openssl.\033[0m"
            )
            logging.info(
                f"{RUST_LOG_PREFIX}\033[93mPausing. To complete install, restart your machine using sudo reboot.\033[0m"
            )
            time.sleep(1e9)

        except subprocess.CalledProcessError as e:
            logging.error(f"{RUST_LOG_PREFIX}Failed to install Rust and Cargo: {e}")
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

    logging.info(f"{RUST_LOG_PREFIX}Installing Rust {TOOLCHAIN}...")
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
        logging.info(
            f"{RUST_LOG_PREFIX}Rust {TOOLCHAIN} has been successfully installed."
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"{RUST_LOG_PREFIX}Failed to install Rust toolchain: {e}")
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
        logging.info(f"{JOLT_LOG_PREFIX}Jolt is already installed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.info(f"{JOLT_LOG_PREFIX}Jolt not found. Installing...")
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
            logging.info(f"{JOLT_LOG_PREFIX}Jolt has been successfully installed.")
        except subprocess.CalledProcessError as e:
            logging.error(f"{JOLT_LOG_PREFIX}Failed to install Jolt: {e}")
            raise RuntimeError(
                "Jolt installation failed. Please install it manually."
            ) from e

    logging.info(f"{JOLT_LOG_PREFIX}Running jolt install-toolchain...")
    try:
        subprocess.run(
            [
                os.path.join(os.path.expanduser("~"), ".cargo", "bin", "jolt"),
                "install-toolchain",
            ],
            check=True,
        )
        logging.info(f"{JOLT_LOG_PREFIX}jolt install-toolchain completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"{JOLT_LOG_PREFIX}Failed to run jolt install-toolchain: {e}")
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
            logging.info(
                JOLT_LOG_PREFIX
                + f"Ignoring model {model_hash} as it is in the ignored list."
            )
            continue

        metadata_file = os.path.join(MODEL_DIR, model_hash, "metadata.json")
        if not os.path.isfile(metadata_file):
            logging.warning(
                f"{JOLT_LOG_PREFIX}Metadata file not found for {model_hash}. Skipping."
            )
            continue

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            logging.error(f"{JOLT_LOG_PREFIX}Failed to parse JSON from {metadata_file}")
            continue

        if metadata.get("proof_system") != "JOLT":
            logging.info(f"{JOLT_LOG_PREFIX}Skipping non-Jolt circuit: {model_hash}")
            continue

        circuit_path = os.path.join(
            MODEL_DIR, model_hash, "target", "release", "circuit"
        )
        if os.path.exists(circuit_path):
            logging.info(f"{JOLT_LOG_PREFIX}Circuit already compiled for {model_hash}")
            continue

        logging.info(f"{JOLT_LOG_PREFIX}Compiling circuit for {model_hash}")
        try:
            subprocess.run(
                ["cargo", "build", "--release"],
                cwd=os.path.join(MODEL_DIR, model_hash),
                check=True,
            )
            logging.info(
                f"{JOLT_LOG_PREFIX}Successfully compiled circuit for {model_hash}"
            )
        except subprocess.CalledProcessError as e:
            logging.error(
                f"{JOLT_LOG_PREFIX}Failed to compile circuit for {model_hash}: {e}"
            )

    logging.info(f"{JOLT_LOG_PREFIX}Jolt circuit compilation process completed.")


def is_safe_path(base_path, path):
    return os.path.realpath(path).startswith(os.path.realpath(base_path))


def safe_extract(tar, path):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_safe_path(path, member_path):
            continue
        tar.extract(member, path)
