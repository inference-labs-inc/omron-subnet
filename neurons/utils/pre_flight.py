import traceback
import os
import subprocess
import time
import json
import requests

# trunk-ignore(pylint/E0611)
from bittensor import logging


LOCAL_SNARKJS_INSTALL_DIR = os.path.join(os.path.expanduser("~"), ".snarkjs")
LOCAL_SNARKJS_PATH = os.path.join(
    LOCAL_SNARKJS_INSTALL_DIR, "node_modules", ".bin", "snarkjs"
)


def run_shared_preflight_checks():
    """
    This function executes a series of checks to ensure the environment is properly
    set up for both validator and miner operations.
    Checks:
    - Model files are synced up
    - Node.js >= 20 is installed
    - SnarkJS is installed

    Raises:
        Exception: If any of the pre-flight checks fail.
    """
    preflight_checks = [
        ("Syncing model files", sync_model_files),
        ("Ensuring Node.js version", ensure_nodejs_version),
        ("Checking SnarkJS installation", ensure_snarkjs_installed),
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


def is_safe_path(base_path, path):
    return os.path.realpath(path).startswith(os.path.realpath(base_path))


def safe_extract(tar, path):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_safe_path(path, member_path):
            continue
        tar.extract(member, path)
