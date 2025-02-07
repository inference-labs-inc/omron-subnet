import os
import json
import bittensor as bt
from constants import MAX_CIRCUIT_SIZE_GB
import traceback


class CircuitValidator:
    REQUIRED_FILES = [
        "vk.key",
        "pk.key",
        "settings.json",
        "model.compiled",
    ]

    REQUIRED_SETTINGS = {
        "run_args": {
            "input_visibility": "Private",
            "output_visibility": "Public",
            "param_visibility": "Private",
            "commitment": "KZG",
        }
    }

    @classmethod
    def validate_files(cls, circuit_dir: str) -> bool:
        """
        Validate that all required files are present and in the correct format.
        """
        try:
            bt.logging.debug(f"Validating circuit files in {circuit_dir}")

            # List all files in the directory
            files = os.listdir(circuit_dir)
            bt.logging.debug(f"Found files: {files}")

            # Required files
            required_files = ["circuit.json", "circuit.wasm", "circuit.zkey", "vk.key"]

            # Check for required files
            for file in required_files:
                if file not in files:
                    bt.logging.error(f"Missing required file: {file}")
                    return False
                bt.logging.debug(f"Found required file: {file}")

            # Validate file contents
            try:
                with open(os.path.join(circuit_dir, "circuit.json"), "r") as f:
                    circuit_json = json.load(f)
                    if not isinstance(circuit_json, dict):
                        bt.logging.error("Invalid circuit.json format")
                        return False
                    bt.logging.debug("circuit.json is valid")
            except json.JSONDecodeError:
                bt.logging.error("Failed to parse circuit.json")
                return False
            except Exception as e:
                bt.logging.error(f"Error reading circuit.json: {e}")
                return False

            # Check file sizes
            for file in required_files:
                size = os.path.getsize(os.path.join(circuit_dir, file))
                if size == 0:
                    bt.logging.error(f"File {file} is empty")
                    return False
                bt.logging.debug(f"File {file} size: {size} bytes")

            bt.logging.success("All circuit files validated successfully")
            return True

        except Exception as e:
            bt.logging.error(f"Error validating circuit files: {e}")
            traceback.print_exc()
            return False

    @classmethod
    def _validate_size(cls, circuit_dir: str) -> bool:
        total_size = sum(
            os.path.getsize(os.path.join(circuit_dir, f))
            for f in os.listdir(circuit_dir)
            if os.path.isfile(os.path.join(circuit_dir, f))
        )
        if total_size > MAX_CIRCUIT_SIZE_GB * 1024 * 1024 * 1024:
            bt.logging.error(
                f"Circuit files too large: {total_size / (1024 * 1024 * 1024):.2f} GB"
            )
            return False
        return True

    @classmethod
    def _validate_required_files(cls, circuit_dir: str) -> bool:
        for f in cls.REQUIRED_FILES:
            if not os.path.exists(os.path.join(circuit_dir, f)):
                bt.logging.error(f"Missing required file: {f}")
                return False
        return True

    @classmethod
    def _validate_settings(cls, circuit_dir: str) -> bool:
        try:
            with open(os.path.join(circuit_dir, "settings.json")) as f:
                settings = json.load(f)
                if "run_args" not in settings:
                    bt.logging.error("Missing run_args in settings.json")
                    return False

                run_args = settings["run_args"]
                required_args = cls.REQUIRED_SETTINGS["run_args"]

                for key, value in required_args.items():
                    if key not in run_args:
                        bt.logging.error(f"Missing required run_args setting: {key}")
                        return False
                    if run_args[key] != value:
                        bt.logging.error(
                            f"Invalid value for {key}: expected {value}, got {run_args[key]}"
                        )
                        return False

            return True
        except json.JSONDecodeError:
            bt.logging.error("Invalid JSON in settings.json")
            return False
