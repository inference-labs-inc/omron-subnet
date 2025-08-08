from __future__ import annotations
import hashlib
import json
import os
from typing import TYPE_CHECKING
import subprocess
import bittensor as bt
import traceback

from execution_layer.proof_handlers.base_handler import ProofSystemHandler
from execution_layer.generic_input import GenericInput
from constants import LOCAL_TEEONNX_PATH
from execution_layer.session_storage import DCAPSessionStorage

if TYPE_CHECKING:
    from execution_layer.verified_model_session import VerifiedModelSession

SGX_ENCLAVE_PATH = "/dev/sgx_enclave"
SGX_PROVISION_PATH = "/dev/sgx_provision"
WORKSPACE_PATH = "/workspace/user"
IMAGE_TAG = "sha-edeb481"
SGX_IMAGE = f"ghcr.io/zkonduit/teeonnx-sgx:{IMAGE_TAG}"
# flake8: noqa: E501
EXPECTED_MRENCLAVE = "[97, 230, 108, 244, 156, 207, 32, 252, 33, 179, 107, 145, 201, 52, 165, 254, 21, 175, 13, 164, 221, 23, 245, 161, 243, 141, 134, 177, 89, 36, 102, 4]"


class DCAPHandler(ProofSystemHandler):
    """
    Handler for the EZKL proof system.
    This class provides methods for generating and verifying proofs using EZKL.
    """

    def gen_input_file(self, session: VerifiedModelSession):
        bt.logging.trace("Generating input file")
        if session.inputs is None:
            raise ValueError("Session inputs cannot be None when generating input file")
        os.makedirs(os.path.dirname(session.session_storage.input_path), exist_ok=True)
        with open(session.session_storage.input_path, "w", encoding="utf-8") as f:
            json.dump(session.inputs.data, f)
        bt.logging.trace(f"Generated input.json with data: {session.inputs.data}")

    def gen_proof(self, session: VerifiedModelSession) -> tuple[str, str]:
        try:
            bt.logging.debug("Starting proof generation...")

            self.generate_witness(session)
            bt.logging.trace("Generating proof")

            result = subprocess.run(
                [
                    LOCAL_TEEONNX_PATH,
                    "prove",
                    "--quote",
                    (
                        session.session_storage.quote_path
                        if isinstance(session.session_storage, DCAPSessionStorage)
                        else ValueError("Session storage is not a DCAPSessionStorage")
                    ),
                    "--proof",
                    session.session_storage.proof_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            bt.logging.trace(
                f"Proof generated: {session.session_storage.proof_path}, result: {result.stdout}"
            )

            with open(session.session_storage.proof_path, "r", encoding="utf-8") as f:
                try:
                    proof = json.load(f)
                except json.JSONDecodeError:
                    bt.logging.error(
                        f"Failed to load proof from {session.session_storage.proof_path}"
                    )
                    raise

            return json.dumps(proof), json.dumps(proof["instances"])

        except Exception as e:
            bt.logging.error(f"An error occurred during proof generation: {e}")
            traceback.print_exc()
            raise

    def verify_proof(
        self,
        session: VerifiedModelSession,
        validator_inputs: GenericInput,
        proof: str | dict,
    ) -> bool:
        if not proof:
            return False

        if isinstance(proof, str):
            proof_json = json.loads(proof)
        else:
            proof_json = proof

        with open(session.session_storage.proof_path, "w", encoding="utf-8") as f:
            json.dump(proof_json, f)

        input_hash = hashlib.sha256(validator_inputs.data).hexdigest()[:64]

        with open(session.model.paths.compiled_model, "rb") as f:
            try:
                model_hash = hashlib.sha256(f.read()).hexdigest()[:64]
            except Exception as e:
                bt.logging.error(f"Failed to hash model: {e}")
                raise

        with open(session.session_storage.witness_path, "rb") as f:
            try:
                witness_hash = hashlib.sha256(f.read()).hexdigest()[:64]
            except Exception as e:
                bt.logging.error(f"Failed to hash witness: {e}")
                raise

        try:
            result = subprocess.run(
                [
                    LOCAL_TEEONNX_PATH,
                    "--verify",
                    session.session_storage.proof_path,
                    "--input-hash",
                    input_hash,
                    "--model-hash",
                    model_hash,
                    "--output-hash",
                    witness_hash,
                    "--mrenclave",
                    EXPECTED_MRENCLAVE,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return "verified: true" in result.stdout
        except subprocess.TimeoutExpired:
            bt.logging.error("Verification process timed out after 60 seconds")
            return False
        except subprocess.CalledProcessError:
            return False

    def generate_witness(
        self, session: VerifiedModelSession, return_content: bool = False
    ) -> list | dict:
        bt.logging.trace("Generating witness")

        result = subprocess.run(
            [
                "docker",
                "run",
                "--device",
                SGX_ENCLAVE_PATH,
                "--device",
                SGX_PROVISION_PATH,
                "-v",
                f"{session.session_storage.base_path}:{WORKSPACE_PATH}",
                SGX_IMAGE,
                "gen-output",
                "--input",
                os.path.join(
                    WORKSPACE_PATH, os.path.basename(session.session_storage.input_path)
                ),
                "--model",
                os.path.join(
                    WORKSPACE_PATH, os.path.basename(session.model.paths.compiled_model)
                ),
                "--output",
                os.path.join(
                    WORKSPACE_PATH,
                    os.path.basename(session.session_storage.witness_path),
                ),
                "--quote",
                os.path.join(
                    WORKSPACE_PATH,
                    os.path.basename(
                        session.session_storage.quote_path
                        if isinstance(session.session_storage, DCAPSessionStorage)
                        else ValueError("Session storage is not a DCAPSessionStorage")
                    ),
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        bt.logging.debug(f"Gen witness result: {result.stdout}")

        if return_content:
            with open(session.session_storage.witness_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return result.stdout
