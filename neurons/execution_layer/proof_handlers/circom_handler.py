import json
import os

# trunk-ignore(bandit/B404)
import subprocess
import traceback

from typing import TYPE_CHECKING
import bittensor as bt
from constants import FIELD_MODULUS
from utils.pre_flight import LOCAL_SNARKJS_PATH
from execution_layer.proof_handlers.base_handler import ProofSystemHandler
from execution_layer.generic_input import GenericInput

if TYPE_CHECKING:
    from execution_layer.verified_model_session import VerifiedModelSession


class CircomHandler(ProofSystemHandler):
    def gen_input_file(self, session):
        bt.logging.trace("Generating input file")

        data = session.inputs.to_json()

        dir_name = os.path.dirname(session.session_storage.input_path)
        os.makedirs(dir_name, exist_ok=True)

        with open(session.session_storage.input_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        bt.logging.trace(f"Generated input.json with data: {data}")

    def gen_proof(self, session):
        try:
            bt.logging.debug(
                f"Starting proof generation with paths: {session.session_storage.input_path}, "
                f"{session.model.paths.compiled_model}, {session.model.paths.pk}, "
                f"{session.session_storage.proof_path}, {session.session_storage.public_path}"
            )

            proof = self.proof_worker(
                input_path=session.session_storage.input_path,
                circuit_path=session.model.paths.compiled_model,
                pk_path=session.model.paths.pk,
                proof_path=session.session_storage.proof_path,
                public_path=session.session_storage.public_path,
            )

            return proof

        except Exception as e:
            bt.logging.error(f"An error occurred during proof generation: {e}")
            raise

    def generate_witness(self, session, return_content: bool = False):
        try:
            bt.logging.debug("Generating witness")
            command = [
                LOCAL_SNARKJS_PATH,
                "wc",
                session.model.paths.compiled_model,
                session.session_storage.input_path,
                session.session_storage.witness_path,
            ]

            # trunk-ignore(bandit/B603)
            result = subprocess.run(command, check=True, capture_output=True, text=True)

            if result.returncode == 0:
                bt.logging.info(
                    f"Generated witness in {session.session_storage.witness_path}"
                )
                if return_content:
                    json_path = os.path.join(
                        session.session_storage.base_path, "witness.json"
                    )
                    # trunk-ignore(bandit/B603)
                    subprocess.run(
                        [
                            LOCAL_SNARKJS_PATH,
                            "wej",
                            session.session_storage.witness_path,
                            json_path,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    return json.load(open(json_path, "r", encoding="utf-8"))
                return session.session_storage.witness_path
            bt.logging.error(f"Failed to generate witness. Error: {result.stderr}")
            bt.logging.error(f"Command output: {result.stdout}")
            raise RuntimeError(f"Witness generation failed: {result.stderr}")
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"Error generating witness: {e}")
            bt.logging.error(f"Command output: {e.stdout}")
            bt.logging.error(f"Command error: {e.stderr}")
            raise RuntimeError(f"Witness generation failed: {str(e)}") from e
        except Exception as e:
            bt.logging.error(f"Unexpected error during witness generation: {e}")
            raise RuntimeError(
                f"Unexpected error during witness generation: {str(e)}"
            ) from e

    def verify_proof(
        self,
        session: "VerifiedModelSession",
        validator_inputs: GenericInput,
        proof: dict,
    ) -> bool:
        try:
            with open(
                session.session_storage.proof_path, "w", encoding="utf-8"
            ) as proof_file:
                json.dump(proof, proof_file)

            # Get public inputs order and sizes from circuit settings
            public_inputs = session.model.settings["public_inputs"]
            input_order = public_inputs["order"]
            input_sizes = public_inputs["sizes"]

            # Replace inputs in public_inputs with session inputs, to ensure
            # the proof was generated against validator-provided inputs
            with open(session.session_storage.input_path, "r", encoding="utf-8") as f:
                session_inputs = json.load(f)

            current_index = 0
            updated_public_data = session_inputs.copy()
            for input_name in input_order:
                if input_name in validator_inputs.to_json():
                    value = validator_inputs.to_json()[input_name]
                    if isinstance(value, list):
                        for i, item in enumerate(value):
                            if i < input_sizes[input_name]:
                                new_value = str(
                                    int(
                                        item if item >= 0 else FIELD_MODULUS - abs(item)
                                    )
                                )
                                updated_public_data[current_index] = new_value
                            current_index += 1
                    else:
                        new_value = value if value >= 0 else FIELD_MODULUS - abs(value)
                        updated_public_data[current_index] = new_value
                        current_index += 1
                else:
                    current_index += input_sizes[input_name]
            with open(
                session.session_storage.public_path, "w", encoding="utf-8"
            ) as public_file:
                json.dump(updated_public_data, public_file)

            bt.logging.trace("Diff between original and updated public_data:")
            for i, (old, new) in enumerate(
                zip(validator_inputs.to_json().values(), updated_public_data)
            ):
                if old != new:
                    bt.logging.trace(f"Index {i}: {old} -> {new}")

            result = subprocess.run(
                [
                    LOCAL_SNARKJS_PATH,
                    "g16v",
                    session.model.paths.vk,
                    session.session_storage.public_path,
                    session.session_storage.proof_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            bt.logging.trace(f"Proof verification stdout: {result.stdout}")
            bt.logging.trace(f"Proof verification stderr: {result.stderr}")
            return "OK!" in result.stdout
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"Proof verification failed: {e}")
            bt.logging.error(f"Proof verification stdout: {e.stdout}")
            bt.logging.error(f"Proof verification stderr: {e.stderr}")
            return False
        except Exception as e:
            bt.logging.error(f"Unexpected error during proof verification: {e}")
            bt.logging.error(f"Error traceback: {traceback.format_exc()}")
            return False

    @staticmethod
    def proof_worker(
        input_path, circuit_path, pk_path, proof_path, public_path
    ) -> tuple[str, str]:
        try:
            # trunk-ignore(bandit/B603)
            result = subprocess.run(
                [
                    LOCAL_SNARKJS_PATH,
                    "g16f",
                    input_path,
                    circuit_path,
                    pk_path,
                    proof_path,
                    public_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            bt.logging.debug(f"Proof generated: {proof_path}")
            bt.logging.trace(f"Proof generation stdout: {result.stdout}")
            bt.logging.trace(f"Proof generation stderr: {result.stderr}")
            proof = None
            with open(proof_path, "r", encoding="utf-8") as proof_file:
                proof = proof_file.read()
            with open(public_path, "r", encoding="utf-8") as public_file:
                public_data = public_file.read()
            return proof, public_data
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"Error generating proof: {e}")
            bt.logging.error(f"Proof generation stdout: {e.stdout}")
            bt.logging.error(f"Proof generation stderr: {e.stderr}")
            raise

    @staticmethod
    def aggregate_proofs(
        session: "VerifiedModelSession", proofs: list[str]
    ) -> tuple[str, float]:
        raise NotImplementedError(
            "Aggregation of proofs is not implemented for CircomHandler"
        )
