from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from execution_layer.verified_model_session import VerifiedModelSession


class ProofSystemHandler(ABC):
    """
    An abstract base class for proof system handlers.
    """

    @abstractmethod
    def gen_input_file(self, session: VerifiedModelSession):
        """
        Generate an input file for the proof system.

        Args:
            session (VerifiedModelSession): The current handler session.
        """

    @abstractmethod
    def gen_proof(self, session: VerifiedModelSession) -> tuple[str, str]:
        """
        Generate a proof for the given session.

        Args:
            session (VerifiedModelSession): The current handler session.

        Returns:
            tuple[str, str]: A tuple containing the proof content (str),
            the public data (str).
        """

    @abstractmethod
    def verify_proof(
        self, session: VerifiedModelSession, public_data: list[str], proof: dict | str
    ) -> bool:
        """
        Verify a proof for the given session.

        Args:
            session (VerifiedModelSession): The current handler session.
            public_data (list[float]): The public data to verify the proof against.
            proof (dict | str): The proof to verify.
        """

    @abstractmethod
    def generate_witness(
        self, session: VerifiedModelSession, return_content: bool = False
    ) -> list | dict:
        """
        Generate a witness for the given session.

        Args:
            session (VerifiedModelSession): The current handler session.
            return_content (bool): Whether to return the witness content.
        """

    @abstractmethod
    def aggregate_proofs(
        self, session: VerifiedModelSession, proofs: list[str]
    ) -> tuple[str, float]:
        """
        Aggregate multiple proofs into a single proof for the given session.

        Returns:
            tuple[str, float]: A tuple containing the aggregated proof content (str)
            and the time taken to aggregate the proofs (float).
        """
