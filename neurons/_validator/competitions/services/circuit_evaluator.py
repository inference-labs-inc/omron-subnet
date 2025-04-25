import os
import json
import time
import tempfile
import subprocess
import torch
import numpy as np
import bittensor as bt
from typing import Tuple, List, Optional
from constants import LOCAL_EZKL_PATH
from _validator.competitions.services.sota_manager import SotaManager
from _validator.competitions.services.data_source import (
    CompetitionDataSource,
    RandomDataSource,
    RemoteDataSource,
)
from utils.wandb_logger import safe_log
import shutil
import traceback
from _validator.competitions.services.data_source import (
    CompetitionDataProcessor,
)
import logging
from utils.system import get_temp_folder
import sys

logging.getLogger("onnxruntime").setLevel(logging.ERROR)
os.environ["ONNXRUNTIME_LOGGING_LEVEL"] = "3"


class CircuitEvaluator:
    def __init__(
        self,
        config: dict,
        competition_directory: str,
        sota_manager: SotaManager,
    ):
        self.config = config
        self.competition_directory = competition_directory
        self.sota_manager = sota_manager

        self.baseline_model = os.path.join(
            self.competition_directory, config["baseline_model_path"]
        )
        bt.logging.debug(f"Using baseline model at: {self.baseline_model}")

        self.is_onnx = not isinstance(self.baseline_model, torch.nn.Module)

        self.onnx_venv = os.path.abspath(
            os.path.join(competition_directory, "onnx_venv")
        )
        self.onnx_runner = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "onnx_runner.py"
        )

        if self.is_onnx:
            if not os.path.exists(self.onnx_venv):
                if not self._setup_onnx_env():
                    raise RuntimeError("Failed to set up ONNX environment")
            else:
                bt.logging.debug(f"Using existing ONNX environment at {self.onnx_venv}")

        self.data_source = self._setup_data_source()

    def _get_python_path(self):
        python_path = os.path.join(self.onnx_venv, "bin", "python")
        if os.path.exists(python_path):
            return python_path

        python_path = os.path.join(self.onnx_venv, "bin", "python3")
        if os.path.exists(python_path):
            return python_path

        for root, dirs, files in os.walk(self.onnx_venv):
            for file in files:
                if file == "python" or file == "python3":
                    return os.path.join(root, file)

        return None

    def _get_pip_path(self):
        pip_path = os.path.join(self.onnx_venv, "bin", "pip")
        if os.path.exists(pip_path):
            return pip_path

        pip_path = os.path.join(self.onnx_venv, "bin", "pip3")
        if os.path.exists(pip_path):
            return pip_path

        for root, dirs, files in os.walk(self.onnx_venv):
            for file in files:
                if file == "pip" or file == "pip3":
                    return os.path.join(root, file)

        return None

    def _find_site_packages(self):
        site_packages = None
        for root, dirs, files in os.walk(self.onnx_venv):
            if os.path.basename(root) == "site-packages":
                site_packages = root
                break

        if not site_packages:
            python_path = self._get_python_path()
            if python_path:
                try:
                    python_version_result = subprocess.run(
                        [python_path, "--version"],
                        capture_output=True,
                        text=True,
                    )
                    python_version_output = python_version_result.stdout.strip()
                    if not python_version_output:
                        python_version_output = python_version_result.stderr.strip()
                    python_version = python_version_output.split()[1][:3]
                    site_packages = os.path.join(
                        self.onnx_venv,
                        "lib",
                        f"python{python_version}",
                        "site-packages",
                    )
                except Exception:
                    site_packages = os.path.join(
                        self.onnx_venv, "lib", "python3.10", "site-packages"
                    )
            else:
                site_packages = os.path.join(
                    self.onnx_venv, "lib", "python3.10", "site-packages"
                )

        if not os.path.exists(site_packages):
            os.makedirs(site_packages, exist_ok=True)

        return site_packages

    def _get_runner_path(self):
        runner_path = None
        for root, dirs, files in os.walk(self.onnx_venv):
            if "onnx_runner.py" in files:
                runner_path = os.path.join(root, "onnx_runner.py")
                break

        if not runner_path or not os.path.exists(runner_path):
            site_packages = self._find_site_packages()
            runner_path = os.path.join(site_packages, "onnx_runner.py")
            if not os.path.exists(runner_path):
                shutil.copy2(self.onnx_runner, runner_path)

        if not os.path.exists(runner_path):
            runner_path = self.onnx_runner

        return runner_path

    def _setup_onnx_env(self):
        try:
            parent_dir = os.path.dirname(self.onnx_venv)
            if not os.path.exists(parent_dir):
                bt.logging.error(f"Parent directory {parent_dir} does not exist")
                return False

            if not os.access(parent_dir, os.W_OK):
                bt.logging.error(f"No write permission for directory {parent_dir}")
                return False

            os.makedirs(self.onnx_venv, exist_ok=True)
            bt.logging.debug(f"Creating ONNX venv at {self.onnx_venv}")

            result = subprocess.run(
                [sys.executable, "-m", "venv", self.onnx_venv],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                bt.logging.error(
                    f"Failed to create venv using {sys.executable}: {result.stderr}"
                )
                return False

            python_path = self._get_python_path()
            if not python_path:
                bt.logging.error(
                    f"Python not found in venv at {self.onnx_venv} after creation"
                )
                return False

            bt.logging.debug(f"Found Python at {python_path}")

            python_version_result = subprocess.run(
                [python_path, "--version"],
                capture_output=True,
                text=True,
            )

            if python_version_result.returncode != 0:
                bt.logging.error(
                    f"Failed to get Python version: {python_version_result.stderr}"
                )
                return False

            python_version_output = python_version_result.stdout.strip()
            if not python_version_output:
                python_version_output = python_version_result.stderr.strip()

            python_version = python_version_output.split()[1][:3]
            bt.logging.debug(f"Python version: {python_version}")

            pip_path = self._get_pip_path()
            if not pip_path:
                bt.logging.error(f"Pip not found in venv at {self.onnx_venv}")
                return False

            bt.logging.debug(f"Found pip at {pip_path}")

            pip_result = subprocess.run(
                [pip_path, "install", "numpy", "onnxruntime==1.20.1"],
                capture_output=True,
                text=True,
            )

            if pip_result.returncode != 0:
                bt.logging.error(f"Failed to install dependencies: {pip_result.stderr}")
                return False

            site_packages = self._find_site_packages()
            bt.logging.debug(f"Using site-packages at {site_packages}")

            runner_path = os.path.join(site_packages, "onnx_runner.py")
            shutil.copy2(self.onnx_runner, runner_path)
            bt.logging.debug(f"Copied onnx_runner.py to {runner_path}")

            bt.logging.success(
                f"ONNX environment set up at {self.onnx_venv} with Python {python_version}"
            )
            return True
        except Exception as e:
            bt.logging.error(f"Error setting up ONNX environment: {str(e)}")
            bt.logging.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def _setup_data_source(self) -> CompetitionDataSource:
        try:
            data_config = self.config.get("data_source", {})
            bt.logging.info(f"Data source config: {data_config}")

            processor = None
            processor_path = os.path.join(
                self.competition_directory, "data_processor.py"
            )
            bt.logging.info(f"Looking for data processor at {processor_path}")
            if os.path.exists(processor_path):
                bt.logging.info("Found data processor, loading module...")
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "data_processor", processor_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, CompetitionDataProcessor)
                        and attr != CompetitionDataProcessor
                    ):
                        processor = attr()
                        bt.logging.info(f"Loaded data processor: {attr.__name__}")
                        break

            if data_config.get("type") == "remote":
                bt.logging.info("Initializing remote data source")
                data_source = RemoteDataSource(
                    self.config, self.competition_directory, processor
                )
                if not data_source.sync_data():
                    bt.logging.error("Failed to sync remote data source")
                    bt.logging.info("Falling back to random data source")
                    return RandomDataSource(
                        self.config, self.competition_directory, processor
                    )
                bt.logging.info("Successfully initialized remote data source")
                return data_source
            bt.logging.info("Using random data source")
            return RandomDataSource(self.config, self.competition_directory, processor)
        except Exception as e:
            bt.logging.error(f"Error setting up data source: {e}")
            traceback.print_exc()
            bt.logging.info("Falling back to random data source due to error")
            return RandomDataSource(self.config, self.competition_directory)

    def _calculate_relative_score(
        self, raw_accuracy: float, proof_size: float, response_time: float
    ) -> float:
        if raw_accuracy == 0:
            return 0.0

        sota_state = self.sota_manager.current_state

        try:
            weights = self.config["evaluation"]["scoring_weights"]
        except Exception as e:
            bt.logging.error(f"Error loading scoring weights, using defaults: {e}")
            weights = {"accuracy": 0.4, "proof_size": 0.3, "response_time": 0.3}

        accuracy_diff = max(0, sota_state.raw_accuracy - raw_accuracy)

        if sota_state.proof_size > 0:
            proof_size_diff = max(
                0, (proof_size - sota_state.proof_size) / sota_state.proof_size
            )
        else:
            proof_size_diff = 0 if proof_size == 0 else 1

        if sota_state.response_time > 0:
            response_time_diff = max(
                0, (response_time - sota_state.response_time) / sota_state.response_time
            )
        else:
            response_time_diff = 0 if response_time == 0 else 1

        total_diff = torch.tensor(
            accuracy_diff * weights["accuracy"]
            + proof_size_diff * weights["proof_size"]
            + response_time_diff * weights["response_time"]
        )

        return torch.exp(-total_diff).item()

    def _calculate_improvements(
        self, raw_accuracy: float, proof_size: float, response_time: float
    ) -> dict:
        sota_state = self.sota_manager.current_state

        if sota_state.sota_relative_score == 0:
            return {
                "raw": {"accuracy": 0, "proof_size": 0, "response_time": 0},
                "weighted": {"accuracy": 0, "proof_size": 0, "response_time": 0},
            }

        try:
            weights = self.config["evaluation"]["scoring_weights"]
        except Exception as e:
            bt.logging.error(f"Error loading scoring weights, using defaults: {e}")
            weights = {"accuracy": 0.4, "proof_size": 0.3, "response_time": 0.3}

        raw_improvements = {
            "accuracy": sota_state.raw_accuracy - raw_accuracy,
            "proof_size": (
                (proof_size - sota_state.proof_size) / sota_state.proof_size
                if sota_state.proof_size > 0
                else 0
            ),
            "response_time": (
                (response_time - sota_state.response_time) / sota_state.response_time
                if sota_state.response_time > 0
                else 0
            ),
        }

        return {
            "raw": raw_improvements,
            "weighted": {
                "accuracy": raw_improvements["accuracy"] * weights["accuracy"],
                "proof_size": raw_improvements["proof_size"] * weights["proof_size"],
                "response_time": raw_improvements["response_time"]
                * weights["response_time"],
            },
        }

    def evaluate(self, circuit_dir: str) -> Tuple[float, float, float, bool, dict]:
        raw_accuracy_scores, proof_sizes, response_times, verification_results = (
            [],
            [],
            [],
            [],
        )
        input_shape = self._get_input_shape(circuit_dir)
        bt.logging.debug(f"Got input shape: {input_shape}")

        if not input_shape:
            bt.logging.error("Failed to get input shape")
            return 0.0, float("inf"), float("inf"), False, {}

        try:
            num_iterations = self.config["evaluation"]["num_iterations"]
        except Exception as e:
            bt.logging.error(f"Error loading num_iterations, using default: {e}")
            num_iterations = 10

        first_valid_output_tensor = None
        all_outputs_identical = True
        successful_valid_outputs = 0

        for i in range(num_iterations):
            bt.logging.debug(f"Running evaluation {i + 1}/{num_iterations}")

            try:
                test_inputs = self.data_source.get_benchmark_data()
                if test_inputs is None:
                    bt.logging.error("Failed to get benchmark data")
                    raw_accuracy_scores.append(0.0)
                    continue

                bt.logging.debug(f"Got benchmark data with shape: {test_inputs.shape}")

                baseline_output = self._run_baseline_model(test_inputs)
                if baseline_output is None:
                    bt.logging.error("Baseline model run failed")
                    raw_accuracy_scores.append(0.0)
                    continue

                bt.logging.debug(
                    f"Got baseline output with shape: {np.array(baseline_output).shape}"
                )

                proof_result = self._generate_proof(circuit_dir, test_inputs)
                if not proof_result:
                    bt.logging.error("Proof generation failed")

                    raw_accuracy_scores.append(0.0)
                    verification_results.append(False)
                    proof_sizes.append(float("inf"))
                    response_times.append(float("inf"))
                    continue

                proof_path, proof_data, response_time = proof_result
                bt.logging.debug(
                    f"Generated proof with size: {len(proof_data['proof'])}"
                )
                response_times.append(response_time)

                proof = proof_data.get("proof", [])
                public_signals = [
                    float(x)
                    for sublist in proof_data.get("pretty_public_inputs", {}).get(
                        "rescaled_outputs", []
                    )
                    for x in sublist
                ]
                proof_sizes.append(len(proof))

                verify_result = self._verify_proof(circuit_dir, proof_path)
                bt.logging.debug(f"Proof verification result: {verify_result}")
                verification_results.append(verify_result)

                if verify_result:
                    current_output_tensor = torch.tensor(public_signals)
                    if first_valid_output_tensor is None:
                        first_valid_output_tensor = current_output_tensor
                        successful_valid_outputs += 1
                    elif all_outputs_identical:
                        successful_valid_outputs += 1
                        if not torch.allclose(
                            first_valid_output_tensor, current_output_tensor, atol=1e-8
                        ):
                            all_outputs_identical = False

                    raw_accuracy = self._compare_outputs(
                        baseline_output, public_signals
                    )
                    bt.logging.debug(f"Raw accuracy: {raw_accuracy}")
                    raw_accuracy_scores.append(raw_accuracy)

                else:
                    bt.logging.error("Proof verification failed")
                    raw_accuracy_scores.append(0.0)

            except Exception as e:
                bt.logging.error(f"Error in evaluation iteration: {str(e)}")
                bt.logging.error(f"Stack trace: {traceback.format_exc()}")

                raw_accuracy_scores.append(0.0)
                verification_results.append(False)
                proof_sizes.append(float("inf"))
                response_times.append(float("inf"))

        if not all(verification_results):
            bt.logging.error(
                "One or more verifications failed - setting all scores to 0"
            )
            return 0.0, float("inf"), float("inf"), False, {}

        if successful_valid_outputs > 1 and all_outputs_identical:
            bt.logging.warning(
                "Detected constant output from circuit across multiple inputs. Penalizing accuracy."
            )
            raw_accuracy_scores = [0.0] * len(raw_accuracy_scores)

        avg_raw_accuracy = (
            sum(raw_accuracy_scores) / len(raw_accuracy_scores)
            if raw_accuracy_scores
            else 0
        )
        avg_proof_size = (
            sum(proof_sizes) / len(proof_sizes) if proof_sizes else float("inf")
        )
        avg_response_time = (
            sum(response_times) / len(response_times)
            if response_times
            else float("inf")
        )

        sota_relative_score = self._calculate_relative_score(
            avg_raw_accuracy, avg_proof_size, avg_response_time
        )

        improvements = self._calculate_improvements(
            avg_raw_accuracy, avg_proof_size, avg_response_time
        )

        safe_log(
            {
                "sota_relative_score": float(sota_relative_score),
                "avg_raw_accuracy": float(avg_raw_accuracy),
                "avg_proof_size": (
                    float(avg_proof_size) if avg_proof_size != float("inf") else -1
                ),
                "avg_response_time": (
                    float(avg_response_time)
                    if avg_response_time != float("inf")
                    else -1
                ),
                "improvements": improvements,
                "total_iterations": num_iterations,
                "successful_iterations": len([s for s in raw_accuracy_scores if s > 0]),
                "verification_success_rate": sum(verification_results)
                / max(len(verification_results), 1),
                "raw_accuracy_distribution": raw_accuracy_scores,
                "proof_sizes_distribution": [
                    float(x) if x != float("inf") else -1 for x in proof_sizes
                ],
                "response_times_distribution": [
                    float(x) if x != float("inf") else -1 for x in response_times
                ],
            }
        )

        bt.logging.info(
            f"Circuit evaluation complete - SOTA Score: {sota_relative_score:.4f}, "
            f"Raw Accuracy: {avg_raw_accuracy:.4f}, "
            f"Proof Size: {avg_proof_size:.0f}, Response Time: {avg_response_time:.2f}s"
        )

        return (
            sota_relative_score,
            avg_proof_size,
            avg_response_time,
            True,
            improvements,
            avg_raw_accuracy,
        )

    def _get_input_shape(self, circuit_dir: str) -> Tuple[int, int] | None:
        try:
            return tuple(self.config["circuit_settings"]["input_shape"])
        except Exception as e:
            bt.logging.error(f"Error reading input shape: {e}")
            return None

    def _run_baseline_model(self, test_inputs: torch.Tensor) -> List | None:
        try:
            if not self.is_onnx:
                return self.baseline_model(test_inputs).tolist()

            python_path = self._get_python_path()
            if not python_path:
                bt.logging.warning(
                    f"Python not found in ONNX venv at {self.onnx_venv}, attempting to recreate"
                )
                if not self._setup_onnx_env():
                    bt.logging.error("Failed to recreate ONNX environment")
                    return None

                python_path = self._get_python_path()
                if not python_path:
                    bt.logging.error(
                        "Still couldn't find Python after recreating environment"
                    )
                    return None

            with (
                tempfile.NamedTemporaryFile(suffix=".npy") as input_file,
                tempfile.NamedTemporaryFile(suffix=".npy") as output_file,
            ):
                np.save(input_file.name, test_inputs.numpy())
                model_path = os.path.abspath(self.baseline_model)

                bt.logging.debug(f"ONNX model path: {model_path}")
                bt.logging.debug(f"Input shape: {test_inputs.shape}")
                bt.logging.debug(f"Input file: {input_file.name}")
                bt.logging.debug(f"Output file: {output_file.name}")
                bt.logging.debug(f"Using Python at: {python_path}")

                if not os.path.exists(model_path):
                    bt.logging.error(f"ONNX model not found at {model_path}")
                    return None

                runner_path = self._get_runner_path()
                bt.logging.debug(f"Using runner at: {runner_path}")

                process = subprocess.Popen(
                    [
                        python_path,
                        runner_path,
                        model_path,
                        input_file.name,
                        output_file.name,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    bt.logging.error(
                        f"ONNX runner failed with code {process.returncode}"
                    )
                    bt.logging.error(f"STDOUT: {stdout}")
                    bt.logging.error(f"STDERR: {stderr}")
                    return None

                output = np.load(output_file.name)
                output_list = output.flatten().tolist()
                return output_list
        except Exception as e:
            bt.logging.error(f"Error running baseline model: {e}")
            bt.logging.error(f"Stack trace: {traceback.format_exc()}")
            return None

    def _generate_proof(
        self, circuit_dir: str, test_inputs: torch.Tensor
    ) -> Tuple[str, dict] | None:
        try:
            input_data = {
                "input_data": [[float(x) for x in test_inputs.flatten().tolist()]]
            }

            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", dir=get_temp_folder(), delete=False
            ) as temp_input:
                json.dump(input_data, temp_input, indent=2)
                temp_input_path = temp_input.name

            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", dir=get_temp_folder(), delete=False
            ) as temp_witness:
                witness_path = temp_witness.name

            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", dir=get_temp_folder(), delete=False
            ) as temp_proof:
                temp_proof_path = temp_proof.name

            model_path = os.path.join(circuit_dir, "model.compiled")
            if not os.path.exists(model_path):
                bt.logging.error(f"model.compiled not found at {model_path}")
                return None

            bt.logging.debug(f"Input data: {json.dumps(input_data, indent=2)}")
            witness_result = subprocess.run(
                [
                    LOCAL_EZKL_PATH,
                    "gen-witness",
                    "--data",
                    temp_input_path,
                    "--compiled-circuit",
                    model_path,
                    "--output",
                    witness_path,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if witness_result.returncode != 0:
                bt.logging.error(
                    f"Witness generation failed with code {witness_result.returncode}"
                )
                bt.logging.error(f"STDOUT: {witness_result.stdout}")
                bt.logging.error(f"STDERR: {witness_result.stderr}")
                return None

            bt.logging.debug("Witness generation successful, starting proof generation")
            proof_start = time.perf_counter()
            prove_result = subprocess.run(
                [
                    LOCAL_EZKL_PATH,
                    "prove",
                    "--compiled-circuit",
                    model_path,
                    "--witness",
                    witness_path,
                    "--pk-path",
                    os.path.join(circuit_dir, "pk.key"),
                    "--proof-path",
                    temp_proof_path,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            proof_time = time.perf_counter() - proof_start

            os.unlink(temp_input_path)
            os.unlink(witness_path)

            if prove_result.returncode != 0:
                bt.logging.error(
                    f"Proof generation failed with code {prove_result.returncode}"
                )
                bt.logging.error(f"STDOUT: {prove_result.stdout}")
                bt.logging.error(f"STDERR: {prove_result.stderr}")
                return None

            with open(temp_proof_path) as f:
                proof_data = json.load(f)
                bt.logging.debug(f"Proof timing - Proof: {proof_time:.3f}s")
                return temp_proof_path, proof_data, proof_time
        except Exception as e:
            bt.logging.error(f"Error generating proof: {e}")
            return None

    def _verify_proof(self, circuit_dir: str, proof_path: str) -> bool:
        try:
            verify_result = subprocess.run(
                [
                    LOCAL_EZKL_PATH,
                    "verify",
                    "--proof-path",
                    proof_path,
                    "--settings-path",
                    os.path.join(circuit_dir, "settings.json"),
                    "--vk-path",
                    os.path.join(circuit_dir, "vk.key"),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            return verify_result.returncode == 0
        except Exception as e:
            bt.logging.error(f"Error verifying proof: {e}")
            return False
        finally:
            if os.path.exists(proof_path):
                os.unlink(proof_path)

    def _compare_outputs(self, expected: list[float], actual: list[float]) -> float:
        try:
            output_shapes = self.config["circuit_settings"]["output_shapes"]
            total_size = sum(np.prod(shape) for shape in output_shapes.values())

            if isinstance(actual, dict) and "pretty_public_inputs" in actual:
                rescaled = actual["pretty_public_inputs"].get("rescaled_outputs", [])
                actual = [float(x) for sublist in rescaled for x in sublist]
            elif isinstance(actual, list):
                if len(actual) > 0 and isinstance(actual[0], list):
                    actual = [float(x) for sublist in actual for x in sublist]

            if len(actual) != total_size:
                bt.logging.error(
                    f"Output size mismatch: expected {total_size}, got {len(actual)}"
                )
                return 0.0

            expected = expected[:total_size]
            expected_tensor = torch.tensor(expected)
            actual_tensor = torch.tensor(actual)

            mae = torch.nn.functional.l1_loss(actual_tensor, expected_tensor)
            raw_accuracy = torch.exp(-mae).item()
            return raw_accuracy
        except Exception as e:
            bt.logging.error(f"Error comparing outputs: {e}")
            bt.logging.error(f"Stack trace: {traceback.format_exc()}")
            return 0.0

    def _run_onnx_model(
        self, model_path: str, inputs: np.ndarray, output_path: str
    ) -> Optional[np.ndarray]:
        try:
            python_path = self._get_python_path()
            if not python_path:
                bt.logging.error(f"Python not found in ONNX venv at {self.onnx_venv}")
                return None

            runner_path = self._get_runner_path()
            input_path = output_path + ".input.npy"
            np.save(input_path, inputs)

            bt.logging.debug(f"Running ONNX model with Python at: {python_path}")
            bt.logging.debug(f"Using runner at: {runner_path}")
            bt.logging.debug(f"Model path: {model_path}")
            bt.logging.debug(f"Input path: {input_path}")
            bt.logging.debug(f"Output path: {output_path}")

            process = subprocess.Popen(
                [python_path, runner_path, model_path, input_path, output_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                bt.logging.error(f"ONNX runner failed with code {process.returncode}")
                bt.logging.error(f"STDOUT: {stdout}")
                bt.logging.error(f"STDERR: {stderr}")
                return None

            try:
                return np.load(output_path)
            except Exception as e:
                bt.logging.error(f"Failed to load ONNX output: {e}")
                return None

        except Exception as e:
            bt.logging.error(f"Error running ONNX model: {e}")
            bt.logging.error(f"Stack trace: {traceback.format_exc()}")
            return None
