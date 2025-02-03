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

ONNX_VENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_venv")
ONNX_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_runner.py")


class CircuitEvaluator:
    def __init__(self, baseline_model: Union[torch.nn.Module, str]):
        self.baseline_model = baseline_model
        self.is_onnx = not isinstance(baseline_model, torch.nn.Module)
        if self.is_onnx and not os.path.exists(ONNX_VENV):
            self._setup_onnx_env()

    def _setup_onnx_env(self):
        subprocess.run(["python", "-m", "venv", ONNX_VENV], check=True)
        pip_path = os.path.join(ONNX_VENV, "bin", "pip")
        subprocess.run(
            [pip_path, "install", "numpy==1.24.3", "onnxruntime==1.17.0"], check=True
        )

    def evaluate(
        self, circuit_dir: str, accuracy_weight: float
    ) -> Tuple[float, float, float, bool]:
        scores, proof_sizes, response_times, verification_results = [], [], [], []
        input_shape = self._get_input_shape(circuit_dir)
        if not input_shape:
            return 0.0, 0.0, 0.0, False

        for i in range(10):
            bt.logging.info(f"Running evaluation {i + 1}/10")
            try:
                test_inputs = torch.randn(*input_shape)
                baseline_output = self._run_baseline_model(test_inputs)
                if baseline_output is None:
                    scores.append(0.0)
                    continue

                start_time = time.time()
                proof_result = self._generate_proof(circuit_dir, test_inputs)
                if not proof_result:
                    scores.append(0.0)
                    continue

                proof_path, proof_data = proof_result
                response_time = time.time() - start_time
                response_times.append(response_time)

                proof = proof_data["proof"]
                public_signals = proof_data["public"]
                proof_sizes.append(len(proof))

                verify_result = self._verify_proof(circuit_dir, proof_path)
                verification_results.append(verify_result)

                if verify_result:
                    raw_score = self._compare_outputs(baseline_output, public_signals)
                    scores.append(raw_score * accuracy_weight)
                else:
                    bt.logging.error("Proof verification failed")
                    scores.append(0.0)

            except Exception as e:
                bt.logging.error(f"Error in evaluation iteration: {e}")
                scores.append(0.0)

        return self._calculate_averages(
            scores, proof_sizes, response_times, verification_results
        )

    def _get_input_shape(self, circuit_dir: str) -> Tuple[int, int] | None:
        try:
            with open(os.path.join(circuit_dir, "settings.json")) as f:
                settings = json.load(f)
                input_shape = settings["input_shape"]
                return tuple(input_shape) if len(input_shape) == 2 else None
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
                result = subprocess.run(
                    [
                        python_path,
                        ONNX_RUNNER,
                        self.baseline_model,
                        input_file.name,
                        output_file.name,
                    ],
                    capture_output=True,
                    text=True,
                )
                return (
                    np.load(output_file.name).tolist()
                    if result.returncode == 0
                    else None
                )
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
            ) as temp_proof:
                temp_proof_path = temp_proof.name

            prove_result = subprocess.run(
                [
                    LOCAL_EZKL_PATH,
                    "prove",
                    "-D",
                    os.path.join(circuit_dir, "model.compiled"),
                    "--input",
                    temp_input_path,
                    "--params",
                    os.path.join(circuit_dir, "settings.json"),
                    "--pk",
                    os.path.join(circuit_dir, "pk.key"),
                    "--proof",
                    temp_proof_path,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            os.unlink(temp_input_path)
            if prove_result.returncode != 0:
                return None

            with open(temp_proof_path) as f:
                return temp_proof_path, json.load(f)
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
        expected_array = torch.tensor(expected).reshape(10, 7)
        actual_array = torch.tensor(actual).reshape(10, 7)

        expected_probs = torch.softmax(expected_array, dim=1)
        actual_probs = torch.softmax(actual_array, dim=1)

        kl_divs = torch.nn.functional.kl_div(
            actual_probs.log(), expected_probs, reduction="none"
        ).sum(dim=1)
        avg_kl = kl_divs.mean().item()

        return 1.0 / (1.0 + avg_kl)

    def _calculate_averages(
        self,
        scores: list[float],
        proof_sizes: list[int],
        response_times: list[float],
        verification_results: list[bool],
    ) -> Tuple[float, float, float, bool]:
        avg_score = sum(scores) / len(scores) if scores else 0
        avg_proof_size = sum(proof_sizes) / len(proof_sizes) if proof_sizes else 0
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )
        verification_success = all(verification_results)

        return avg_score, avg_proof_size, avg_response_time, verification_success
