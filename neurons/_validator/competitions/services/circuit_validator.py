import os
import json
import subprocess
import bittensor as bt
from constants import LOCAL_EZKL_PATH, MAX_CIRCUIT_SIZE_GB


class CircuitValidator:
    REQUIRED_FILES = [
        "vk.key",
        "pk.key",
        "settings.json",
        "model.compiled",
        "network.onnx",
        "input.json",
    ]

    REQUIRED_SETTINGS = [
        "model_path",
        "input_shape",
        "output_shape",
        "param_visibility",
        "scale",
        "bits",
    ]

    @classmethod
    def validate_files(cls, circuit_dir: str) -> bool:
        try:
            if not cls._validate_size(circuit_dir):
                return False

            if not cls._validate_required_files(circuit_dir):
                return False

            if not cls._validate_settings(circuit_dir):
                return False

            if not cls._validate_input_format(circuit_dir):
                return False

            if not cls._validate_ezkl_setup(circuit_dir):
                return False

            return True

        except Exception as e:
            bt.logging.error(f"Error validating circuit files: {e}")
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
                if not all(k in settings for k in cls.REQUIRED_SETTINGS):
                    bt.logging.error("Invalid settings.json format")
                    return False
            return True
        except json.JSONDecodeError:
            bt.logging.error("Invalid JSON in settings.json")
            return False

    @classmethod
    def _validate_input_format(cls, circuit_dir: str) -> bool:
        try:
            with open(os.path.join(circuit_dir, "input.json")) as f:
                input_data = json.load(f)
                if "input_data" not in input_data or not isinstance(
                    input_data["input_data"], list
                ):
                    bt.logging.error("Invalid input.json format")
                    return False
            return True
        except json.JSONDecodeError:
            bt.logging.error("Invalid JSON in input.json")
            return False

    @classmethod
    def _validate_ezkl_setup(cls, circuit_dir: str) -> bool:
        result = subprocess.run(
            [
                LOCAL_EZKL_PATH,
                "setup",
                "--settings-path",
                os.path.join(circuit_dir, "settings.json"),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            bt.logging.error(f"EZKL setup validation failed: {result.stderr}")
            return False
        return True
