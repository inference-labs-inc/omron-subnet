import os
import json
import time
import tempfile
import subprocess
import torch
import numpy as np
import bittensor as bt
from typing import Tuple, Union, List
from constants import LOCAL_EZKL_PATH, TEMP_FOLDER
from _validator.competitions.services.sota_manager import SotaManager
from _validator.competitions.services.data_source import (
    CompetitionDataSource,
    RandomDataSource,
    RemoteDataSource,
)
from utils.wandb_logger import safe_log
import shutil
import traceback


class CircuitEvaluator:
    def __init__(
        self,
        baseline_model: Union[torch.nn.Module, str],
        competition_directory: str,
        sota_manager: SotaManager,
    ):
        self.baseline_model = baseline_model
        self.competition_directory = competition_directory
        self.sota_manager = sota_manager
        self.is_onnx = not isinstance(baseline_model, torch.nn.Module)

        self.onnx_venv = os.path.abspath(
            os.path.join(competition_directory, "onnx_venv")
        )
        self.onnx_runner = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "onnx_runner.py"
        )

        if self.is_onnx and not os.path.exists(self.onnx_venv):
            self._setup_onnx_env()

        self.data_source = self._setup_data_source()

    def _setup_onnx_env(self):
        os.makedirs(self.onnx_venv, exist_ok=True)
        subprocess.run(["python", "-m", "venv", self.onnx_venv], check=True)
        pip_path = os.path.join(self.onnx_venv, "bin", "pip")
        python_path = os.path.join(self.onnx_venv, "bin", "python")

        version_result = subprocess.run(
            [
                python_path,
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        python_version = version_result.stdout.strip()
        bt.logging.debug(f"ONNX venv Python version: {python_version}")

        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        subprocess.run([pip_path, "install", "numpy", "onnxruntime"], check=True)

        site_packages = os.path.join(
            self.onnx_venv, "lib", f"python{python_version}", "site-packages"
        )
        os.makedirs(site_packages, exist_ok=True)
        shutil.copy2(self.onnx_runner, os.path.join(site_packages, "onnx_runner.py"))
        bt.logging.success(
            f"ONNX environment set up at {self.onnx_venv} with Python {python_version}"
        )

    def _setup_data_source(self) -> CompetitionDataSource:
        try:
            config_path = os.path.join(
                self.competition_directory, "competition_config.json"
            )
            with open(config_path) as f:
                config = json.load(f)
                data_config = config.get("data_source", {})
                if data_config.get("type") == "remote":
                    return RemoteDataSource(self.competition_directory)
        except Exception as e:
            bt.logging.error(f"Error setting up data source: {e}")

        return RandomDataSource(self.competition_directory)

    def _calculate_relative_score(
        self, accuracy: float, proof_size: float, response_time: float
    ) -> float:
        sota_state = self.sota_manager.current_state

        try:
            with open(
                os.path.join(self.competition_directory, "competition_config.json")
            ) as f:
                config = json.load(f)
                weights = config["evaluation"]["scoring_weights"]
        except Exception as e:
            bt.logging.error(f"Error loading scoring weights, using defaults: {e}")
            weights = {"accuracy": 0.4, "proof_size": 0.3, "response_time": 0.3}

        accuracy_diff = max(0, sota_state.accuracy - accuracy)

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

    def evaluate(
        self, circuit_dir: str, accuracy_weight: float
    ) -> Tuple[float, float, float, bool]:
        scores, proof_sizes, response_times, verification_results = [], [], [], []
        input_shape = self._get_input_shape(circuit_dir)
        bt.logging.debug(f"Got input shape: {input_shape}")

        safe_log(
            {
                "circuit_eval_status": "started",
                "input_shape": input_shape if input_shape else None,
            }
        )

        if not input_shape:
            bt.logging.error("Failed to get input shape")
            safe_log(
                {
                    "circuit_eval_status": "failed",
                    "error": "Failed to get input shape",
                }
            )
            return 0.0, float("inf"), float("inf"), False

        try:
            with open(
                os.path.join(self.competition_directory, "competition_config.json")
            ) as f:
                config = json.load(f)
                num_iterations = config["evaluation"]["num_iterations"]
        except Exception as e:
            bt.logging.error(f"Error loading num_iterations, using default: {e}")
            num_iterations = 10
            safe_log(
                {
                    "circuit_eval_status": "config_error",
                    "error": str(e),
                    "using_default_iterations": num_iterations,
                }
            )

        for i in range(num_iterations):
            iteration_start = time.time()
            bt.logging.debug(f"Running evaluation {i + 1}/{num_iterations}")
            safe_log(
                {
                    "circuit_eval_status": "iteration_started",
                    "iteration": i + 1,
                    "total_iterations": num_iterations,
                }
            )

            try:
                test_inputs = self.data_source.get_benchmark_data()
                if test_inputs is None:
                    bt.logging.error("Failed to get benchmark data")
                    safe_log(
                        {
                            "circuit_eval_status": "iteration_error",
                            "iteration": i + 1,
                            "error": "Failed to get benchmark data",
                        }
                    )
                    scores.append(0.0)
                    continue

                bt.logging.debug(f"Got benchmark data with shape: {test_inputs.shape}")
                safe_log(
                    {
                        "circuit_eval_status": "got_benchmark_data",
                        "iteration": i + 1,
                        "input_shape": list(test_inputs.shape),
                    }
                )

                baseline_output = self._run_baseline_model(test_inputs)
                if baseline_output is None:
                    bt.logging.error("Baseline model run failed")
                    safe_log(
                        {
                            "circuit_eval_status": "iteration_error",
                            "iteration": i + 1,
                            "error": "Baseline model run failed",
                        }
                    )
                    scores.append(0.0)
                    continue

                bt.logging.debug(
                    f"Got baseline output with shape: {np.array(baseline_output).shape}"
                )
                safe_log(
                    {
                        "circuit_eval_status": "baseline_complete",
                        "iteration": i + 1,
                        "output_shape": list(np.array(baseline_output).shape),
                    }
                )

                proof_result = self._generate_proof(circuit_dir, test_inputs)
                if not proof_result:
                    bt.logging.error("Proof generation failed")
                    safe_log(
                        {
                            "circuit_eval_status": "iteration_error",
                            "iteration": i + 1,
                            "error": "Proof generation failed",
                        }
                    )
                    scores.append(0.0)
                    verification_results.append(False)
                    proof_sizes.append(float("inf"))
                    response_times.append(float("inf"))
                    continue

                proof_path, proof_data, response_time = proof_result
                bt.logging.debug(
                    f"Generated proof with size: {len(proof_data['proof'])}"
                )
                response_times.append(response_time)

                safe_log(
                    {
                        "circuit_eval_status": "proof_generated",
                        "iteration": i + 1,
                        "proof_size": len(proof_data["proof"]),
                        "response_time": response_time,
                    }
                )

                proof = proof_data.get("proof", [])
                public_signals = [
                    float(x)
                    for sublist in proof_data.get("pretty_public_inputs", {}).get(
                        "rescaled_outputs", []
                    )
                    for x in sublist
                ]
                bt.logging.info(f"Extracted public signals: {public_signals}")
                proof_sizes.append(len(proof))

                verify_result = self._verify_proof(circuit_dir, proof_path)
                bt.logging.debug(f"Proof verification result: {verify_result}")
                verification_results.append(verify_result)

                safe_log(
                    {
                        "circuit_eval_status": "proof_verified",
                        "iteration": i + 1,
                        "verification_success": verify_result,
                    }
                )

                if verify_result:
                    accuracy_score = self._compare_outputs(
                        baseline_output, public_signals
                    )
                    bt.logging.debug(f"Accuracy score: {accuracy_score}")
                    scores.append(accuracy_score)

                    safe_log(
                        {
                            "circuit_eval_status": "iteration_complete",
                            "iteration": i + 1,
                            "accuracy_score": float(accuracy_score),
                            "proof_size": len(proof),
                            "response_time": response_time,
                            "iteration_duration": time.time() - iteration_start,
                        }
                    )
                else:
                    bt.logging.error("Proof verification failed")
                    safe_log(
                        {
                            "circuit_eval_status": "iteration_error",
                            "iteration": i + 1,
                            "error": "Proof verification failed",
                        }
                    )
                    scores.append(0.0)

            except Exception as e:
                bt.logging.error(f"Error in evaluation iteration: {str(e)}")
                bt.logging.error(f"Stack trace: {traceback.format_exc()}")
                safe_log(
                    {
                        "circuit_eval_status": "iteration_error",
                        "iteration": i + 1,
                        "error": str(e),
                    }
                )
                scores.append(0.0)
                verification_results.append(False)
                proof_sizes.append(float("inf"))
                response_times.append(float("inf"))

        if not all(verification_results):
            bt.logging.error(
                "One or more verifications failed - setting all scores to 0"
            )
            safe_log(
                {
                    "circuit_eval_status": "eval_failed",
                    "error": "One or more verifications failed",
                    "verification_results": verification_results,
                }
            )
            return 0.0, float("inf"), float("inf"), False

        avg_accuracy = sum(scores) / len(scores) if scores else 0
        avg_proof_size = (
            sum(proof_sizes) / len(proof_sizes) if proof_sizes else float("inf")
        )
        avg_response_time = (
            sum(response_times) / len(response_times)
            if response_times
            else float("inf")
        )

        final_score = self._calculate_relative_score(
            avg_accuracy, avg_proof_size, avg_response_time
        )

        safe_log(
            {
                "circuit_eval_status": "eval_complete",
                "final_score": float(final_score),
                "avg_accuracy": float(avg_accuracy),
                "avg_proof_size": (
                    float(avg_proof_size) if avg_proof_size != float("inf") else -1
                ),
                "avg_response_time": (
                    float(avg_response_time)
                    if avg_response_time != float("inf")
                    else -1
                ),
                "total_iterations": num_iterations,
                "successful_iterations": len([s for s in scores if s > 0]),
                "verification_success_rate": sum(verification_results)
                / max(len(verification_results), 1),
                "scores_distribution": scores,
                "proof_sizes_distribution": [
                    float(x) if x != float("inf") else -1 for x in proof_sizes
                ],
                "response_times_distribution": [
                    float(x) if x != float("inf") else -1 for x in response_times
                ],
            }
        )

        bt.logging.info(
            f"Circuit evaluation complete - Score: {final_score:.4f}, Accuracy: {avg_accuracy:.4f}, "
            f"Proof Size: {avg_proof_size:.0f}, Response Time: {avg_response_time:.2f}s"
        )
        return final_score, avg_proof_size, avg_response_time, True

    def _get_input_shape(self, circuit_dir: str) -> Tuple[int, int] | None:
        try:
            config_path = os.path.join(
                self.competition_directory, "competition_config.json"
            )
            bt.logging.debug(f"Reading config from: {config_path}")
            with open(config_path) as f:
                config = json.load(f)
                if (
                    "circuit_settings" in config
                    and "input_shape" in config["circuit_settings"]
                ):
                    return tuple(config["circuit_settings"]["input_shape"])
            return None
        except Exception as e:
            bt.logging.error(f"Error reading input shape: {e}")
            return None

    def _run_baseline_model(self, test_inputs: torch.Tensor) -> List | None:
        try:
            if not self.is_onnx:
                return self.baseline_model(test_inputs).tolist()

            with (
                tempfile.NamedTemporaryFile(suffix=".npy") as input_file,
                tempfile.NamedTemporaryFile(suffix=".npy") as output_file,
            ):
                np.save(input_file.name, test_inputs.numpy())
                python_path = os.path.join(self.onnx_venv, "bin", "python")
                model_path = os.path.abspath(self.baseline_model)

                version_result = subprocess.run(
                    [
                        python_path,
                        "-c",
                        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                python_version = version_result.stdout.strip()
                runner_path = os.path.join(
                    self.onnx_venv,
                    "lib",
                    f"python{python_version}",
                    "site-packages",
                    "onnx_runner.py",
                )

                bt.logging.info(
                    f"Running ONNX model: {model_path} with runner: {runner_path}"
                )
                result = subprocess.run(
                    [
                        python_path,
                        runner_path,
                        model_path,
                        input_file.name,
                        output_file.name,
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    bt.logging.error(f"ONNX runner failed: {result.stderr}")
                    return None

                output = np.load(output_file.name)
                bt.logging.info(f"Raw ONNX output shape: {output.shape}")
                bt.logging.info(f"Raw ONNX output: {output}")

                # Flatten and convert to list
                output_list = output.flatten().tolist()
                bt.logging.info(f"Flattened ONNX output: {output_list}")
                return output_list
        except Exception as e:
            bt.logging.error(f"Error running baseline model: {e}")
            return None

    def _generate_proof(
        self, circuit_dir: str, test_inputs: torch.Tensor
    ) -> Tuple[str, dict] | None:
        try:
            input_data = {
                "input_data": [[float(x) for x in test_inputs.flatten().tolist()]]
            }

            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", dir=TEMP_FOLDER, delete=False
            ) as temp_input:
                json.dump(input_data, temp_input, indent=2)
                temp_input_path = temp_input.name

            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", dir=TEMP_FOLDER, delete=False
            ) as temp_witness:
                witness_path = temp_witness.name

            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", dir=TEMP_FOLDER, delete=False
            ) as temp_proof:
                temp_proof_path = temp_proof.name

            model_path = os.path.join(circuit_dir, "model.compiled")
            bt.logging.info(f"Checking model path: {model_path}")
            if not os.path.exists(model_path):
                bt.logging.error(f"model.compiled not found at {model_path}")
                bt.logging.info("Circuit directory contents:")
                for root, dirs, files in os.walk(circuit_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        bt.logging.info(
                            f"- {file} ({os.path.getsize(file_path)} bytes)"
                        )
                return None

            bt.logging.info(
                f"Running witness generation with input shape: {test_inputs.shape}"
            )
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
                safe_log(
                    {
                        "circuit_eval_status": "witness_gen_failed",
                        "error_code": witness_result.returncode,
                        "stdout": witness_result.stdout,
                        "stderr": witness_result.stderr,
                    }
                )
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
                safe_log(
                    {
                        "circuit_eval_status": "proof_gen_failed",
                        "error_code": prove_result.returncode,
                        "stdout": prove_result.stdout,
                        "stderr": prove_result.stderr,
                    }
                )
                return None

            with open(temp_proof_path) as f:
                proof_data = json.load(f)
                bt.logging.info(
                    f"Proof data structure: {json.dumps(proof_data, indent=2)}"
                )
                bt.logging.info(f"Proof data keys: {list(proof_data.keys())}")
                if "pretty_public_inputs" in proof_data:
                    bt.logging.info(
                        f"Pretty public inputs: {json.dumps(proof_data['pretty_public_inputs'], indent=2)}"
                    )
                    if "rescaled_outputs" in proof_data["pretty_public_inputs"]:
                        bt.logging.info(
                            f"Rescaled outputs: "
                            f"{json.dumps(proof_data['pretty_public_inputs']['rescaled_outputs'], indent=2)}"
                        )
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
            with open(
                os.path.join(self.competition_directory, "competition_config.json")
            ) as f:
                config = json.load(f)
                output_shapes = config["circuit_settings"]["output_shapes"]
                total_size = sum(
                    shape[0] * shape[1] for shape in output_shapes.values()
                )
                bt.logging.info(f"Expected total output size: {total_size}")
                bt.logging.info(f"Expected output: {expected}")
                bt.logging.info(f"Raw actual output: {actual}")

            if isinstance(actual, dict) and "pretty_public_inputs" in actual:
                rescaled = actual["pretty_public_inputs"].get("rescaled_outputs", [])
                bt.logging.info(f"Found rescaled outputs in dict: {rescaled}")
                actual = [float(x) for sublist in rescaled for x in sublist]
            elif isinstance(actual, list):
                if len(actual) > 0 and isinstance(actual[0], list):
                    bt.logging.info(f"Found nested list structure: {actual}")
                    actual = [float(x) for sublist in actual for x in sublist]

            bt.logging.info(f"Processed actual output: {actual}")

            if len(actual) != total_size:
                bt.logging.error(
                    f"Output size mismatch: expected {total_size}, got {len(actual)}"
                )
                return 0.0

            expected = expected[:total_size]
            bt.logging.info(f"Using expected values: {expected}")

            expected_tensor = torch.tensor(expected)
            actual_tensor = torch.tensor(actual)

            mae = torch.nn.functional.l1_loss(actual_tensor, expected_tensor)
            accuracy = torch.exp(-mae).item()
            bt.logging.info(f"MAE: {mae.item()}, Accuracy: {accuracy}")

            return accuracy
        except Exception as e:
            bt.logging.error(f"Error comparing outputs: {e}")
            bt.logging.error(f"Stack trace: {traceback.format_exc()}")
            return 0.0
