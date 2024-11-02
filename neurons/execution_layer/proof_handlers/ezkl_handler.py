from __future__ import annotations
import json
import os
from typing import TYPE_CHECKING
import ezkl
import bittensor as bt

from execution_layer.proof_handlers.base_handler import ProofSystemHandler

if TYPE_CHECKING:
    from execution_layer.verified_model_session import VerifiedModelSession


class EZKLHandler(ProofSystemHandler):
    """
    Handler for the EZKL (Easy Zero-Knowledge Learning) proof system.
    This class provides methods for generating and verifying proofs using EZKL.
    """

    def gen_input_file(self, session: VerifiedModelSession):
        """
        Generate an input file for the EZKL proof system.

        Args:
            session (VerifiedModelSession): The session object containing input data and storage information.
        """
        bt.logging.trace("Generating input file")
        data = {"input_data": session.inputs}
        os.makedirs(os.path.dirname(session.session_storage.input_path), exist_ok=True)
        with open(session.session_storage.input_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        bt.logging.trace(f"Generated input.json with data: {data}")

    def gen_proof(self, session: VerifiedModelSession):
        """
        Generate a proof using the EZKL proof system.

        Args:
            session (VerifiedModelSession): The session object containing necessary paths and data.

        Returns:
            str: The content of the generated proof.
        """
        self.gen_input_file(session)
        bt.logging.trace("Generating witness")
        self.gen_witness(
            session.session_storage.input_path,
            session.model.paths.compiled_model,
            session.session_storage.witness_path,
            session.model.paths.vk,
            session.model.paths.srs,
        )
        bt.logging.trace("Generating proof")
        res = ezkl.prove(  # type: ignore
            session.session_storage.witness_path,
            session.model.paths.compiled_model,
            session.model.paths.pk,
            session.session_storage.proof_path,
            "single",
            session.model.paths.srs,
        )
        bt.logging.trace(
            f"Proof generated: {session.session_storage.proof_path}, result: {res}"
        )
        with open(session.session_storage.proof_path, "r", encoding="utf-8") as f:
            return f.read()

    def verify_proof(self, session: VerifiedModelSession):
        """
        Verify a proof using the EZKL proof system.

        Args:
            session (VerifiedModelSession): The session object containing necessary paths for verification.

        Returns:
            bool: The result of the verification process.
        """
        res = ezkl.verify(  # type: ignore
            session.session_storage.proof_path,
            session.model.paths.settings,
            session.model.paths.vk,
            session.model.paths.srs,
        )
        return res

    def generate_witness(self, session: VerifiedModelSession):
        """
        Generate a witness for the EZKL proof system.

        Args:
            session (VerifiedModelSession): The session object containing necessary paths for witness generation.

        Returns:
            The result of the witness generation process.
        """
        return EZKLHandler.gen_witness(
            session.session_storage.input_path,
            session.model.paths.compiled_model,
            session.session_storage.witness_path,
            session.model.paths.vk,
            session.model.paths.srs,
        )

    @staticmethod
    def gen_witness(input_path, circuit_path, witness_path, vk_path, srs_path):
        """
        Generate a witness for the EZKL proof system.

        Args:
            input_path (str): Path to the input file.
            circuit_path (str): Path to the circuit file.
            witness_path (str): Path to store the generated witness.
            vk_path (str): Path to the verification key.
            srs_path (str): Path to the SRS (Structured Reference String) file.

        Returns:
            The result of the witness generation process.
        """
        bt.logging.trace("Generating witness")
        res = ezkl.gen_witness(input_path, circuit_path, witness_path, vk_path, srs_path)  # type: ignore
        bt.logging.trace(f"Gen witness result: {res}")
        return res
