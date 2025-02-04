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

ONNX_VENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_venv")
ONNX_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_runner.py")


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
        if self.is_onnx and not os.path.exists(ONNX_VENV):
            self._setup_onnx_env()

    def _setup_onnx_env(self):
        subprocess.run(["python", "-m", "venv", ONNX_VENV], check=True)
        pip_path = os.path.join(ONNX_VENV, "bin", "pip")
        subprocess.run(
            [pip_path, "install", "numpy==1.24.3", "onnxruntime==1.17.0"], check=True
        )

    def _calculate_relative_score(
        self, accuracy: float, proof_size: float, response_time: float
    ) -> float:
        sota_state = self.sota_manager.current_state

        accuracy_diff = max(0, sota_state.accuracy - accuracy)
        proof_size_diff = max(
            0, (proof_size - sota_state.proof_size) / sota_state.proof_size
        )
        response_time_diff = max(
            0, (response_time - sota_state.response_time) / sota_state.response_time
        )

        total_diff = torch.tensor(
            accuracy_diff * 0.4 + proof_size_diff * 0.3 + response_time_diff * 0.3
        )

        return torch.exp(-total_diff).item()

    def evaluate(
        self, circuit_dir: str, accuracy_weight: float
    ) -> Tuple[float, float, float, bool]:
        scores, proof_sizes, response_times, verification_results = [], [], [], []
        input_shape = self._get_input_shape(circuit_dir)
        bt.logging.info(f"Got input shape: {input_shape}")
        if not input_shape:
            bt.logging.error("Failed to get input shape")
            return 0.0, 0.0, 0.0, False

        for i in range(10):
            bt.logging.info(f"Running evaluation {i + 1}/10")
            try:
                test_inputs = torch.randn(*input_shape)
                bt.logging.info(
                    f"Generated test inputs with shape: {test_inputs.shape}"
                )

                baseline_output = self._run_baseline_model(test_inputs)
                if baseline_output is None:
                    bt.logging.error("Baseline model run failed")
                    scores.append(0.0)
                    continue
                bt.logging.info(
                    f"Got baseline output with shape: {np.array(baseline_output).shape}"
                )

                start_time = time.time()
                proof_result = self._generate_proof(circuit_dir, test_inputs)
                if not proof_result:
                    bt.logging.error("Proof generation failed")
                    scores.append(0.0)
                    continue

                proof_path, proof_data = proof_result
                bt.logging.info(
                    f"Generated proof with size: {len(proof_data['proof'])}"
                )
                response_time = time.time() - start_time
                response_times.append(response_time)

                proof = proof_data.get("proof", [])
                public_signals = [
                    float(input)
                    for input in proof_data.get("pretty_public_inputs", {}).get(
                        "rescaled_outputs", [[]]
                    )[0]
                ]
                bt.logging.info(f"Public signals: {public_signals}")
                proof_sizes.append(len(proof))

                verify_result = self._verify_proof(circuit_dir, proof_path)
                bt.logging.info(f"Proof verification result: {verify_result}")
                verification_results.append(verify_result)

                if verify_result:
                    accuracy_score = self._compare_outputs(
                        baseline_output, public_signals
                    )
                    bt.logging.info(f"Accuracy score: {accuracy_score}")
                    scores.append(accuracy_score)
                else:
                    bt.logging.error("Proof verification failed")
                    scores.append(0.0)

            except Exception as e:
                bt.logging.error(f"Error in evaluation iteration: {str(e)}")
                scores.append(0.0)

        if not all(verification_results):
            bt.logging.error(
                "One or more verifications failed - setting all scores to 0"
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

        bt.logging.info(
            f"Final metrics - Accuracy: {avg_accuracy}, Proof Size: {avg_proof_size}, "
            f"Response Time: {avg_response_time}, Relative Score: {final_score}"
        )
        return final_score, avg_proof_size, avg_response_time, True

    def _get_input_shape(self, circuit_dir: str) -> Tuple[int, int] | None:
        try:
            config_path = os.path.join(
                self.competition_directory, "competition_config.json"
            )
            bt.logging.info(f"Reading config from: {config_path}")
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
                python_path = os.path.join(ONNX_VENV, "bin", "python")
                model_path = os.path.abspath(self.baseline_model)
                bt.logging.info(f"Running ONNX model: {model_path}")
                result = subprocess.run(
                    [
                        python_path,
                        ONNX_RUNNER,
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
                return np.load(output_file.name).tolist()
        except Exception as e:
            bt.logging.error(f"Error running baseline model: {e}")
            return None

    def _generate_proof(
        self, circuit_dir: str, test_inputs: torch.Tensor
    ) -> Tuple[str, dict] | None:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", dir=TEMP_FOLDER, delete=False
            ) as temp_input:
                json.dump({"input_data": test_inputs.tolist()}, temp_input)
                temp_input_path = temp_input.name

            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", dir=TEMP_FOLDER, delete=False
            ) as temp_witness:
                witness_path = temp_witness.name

            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", dir=TEMP_FOLDER, delete=False
            ) as temp_proof:
                temp_proof_path = temp_proof.name

            witness_result = subprocess.run(
                [
                    LOCAL_EZKL_PATH,
                    "gen-witness",
                    "--data",
                    temp_input_path,
                    "--compiled-circuit",
                    os.path.join(circuit_dir, "model.compiled"),
                    "--output",
                    witness_path,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if witness_result.returncode != 0:
                bt.logging.error(f"Witness generation failed: {witness_result.stderr}")
                return None

            prove_result = subprocess.run(
                [
                    LOCAL_EZKL_PATH,
                    "prove",
                    "--compiled-circuit",
                    os.path.join(circuit_dir, "model.compiled"),
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

            os.unlink(temp_input_path)
            os.unlink(witness_path)

            if prove_result.returncode != 0:
                bt.logging.error(f"Proof generation failed: {prove_result.stderr}")
                return None

            with open(temp_proof_path) as f:
                proof_data = json.load(f)
                bt.logging.info(f"Proof data keys: {list(proof_data.keys())}")
                bt.logging.info(f"Full proof data: {proof_data}")
                return temp_proof_path, proof_data
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
            expected_array = torch.tensor(expected).reshape(1, 6)
            actual_array = torch.tensor(actual).reshape(1, 6)
            bt.logging.info(
                f"Comparing expected {expected_array} with actual {actual_array}"
            )

            mse = torch.nn.functional.mse_loss(actual_array, expected_array)
            accuracy = torch.exp(-mse).item()
            bt.logging.info(f"MSE: {mse.item()}, Accuracy: {accuracy}")

            return accuracy
        except Exception as e:
            bt.logging.error(f"Error comparing outputs: {e}")
            return 0.0
