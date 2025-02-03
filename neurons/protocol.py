from __future__ import annotations
from typing import Dict, Optional

import bittensor as bt

from execution_layer.circuit import ProofSystem


class QueryZkProof(bt.Synapse):
    """
    QueryZkProof class inherits from bt.Synapse.
    It is used to query zkproof of certain model.
    """

    # Required request input, filled by sending dendrite caller.
    query_input: Optional[Dict] = None

    # Optional request output, filled by receiving axon.
    query_output: Optional[str] = None

    def deserialize(self: QueryZkProof) -> str | None:
        """
        unpack query_output
        """
        return self.query_output


class QueryForProvenInference(bt.Synapse):
    """
    A Synapse for querying proven inferences.
    DEV: This synapse is a placeholder.
    """

    query_input: Optional[dict] = None
    query_output: Optional[dict] = None

    def deserialize(self) -> dict | None:
        """
        Deserialize the query_output into a dictionary.
        """
        return self.query_output


class ProofOfWeightsSynapse(bt.Synapse):
    """
    A synapse for conveying proof of weights messages
    """

    subnet_uid: int = 2
    verification_key_hash: str
    proof_system: ProofSystem = ProofSystem.CIRCOM
    inputs: dict
    proof: str
    public_signals: str

    def deserialize(self) -> dict | None:
        """
        Return the proof
        """
        return {
            "inputs": self.inputs,
            "proof": self.proof,
            "public_signals": self.public_signals,
        }


class Competition(bt.Synapse):
    """
    A synapse for conveying competition messages and circuit files
    """

    id: int  # Competition ID
    hash: str  # Circuit hash
    file_name: str  # Name of file being requested
    file_content: Optional[str] = None  # Hex encoded file content
    commitment: Optional[str] = None  # Circuit commitment data from miner
    error: Optional[str] = None  # Error message if something goes wrong

    def deserialize(self) -> dict:
        """Return all fields including required ones"""
        return {
            "id": self.id,
            "hash": self.hash,
            "file_name": self.file_name,
            "file_content": self.file_content,
            "commitment": self.commitment,
            "error": self.error,
        }


class QueryForProofAggregation(bt.Synapse):
    """
    Query for aggregation of multiple proofs into a single proof
    """

    proofs: list[str] = []
    model_id: str or int
    aggregation_proof: Optional[str] = None

    def deserialize(self) -> str | None:
        """
        Return the aggregation proof
        """
        return self.aggregation_proof
