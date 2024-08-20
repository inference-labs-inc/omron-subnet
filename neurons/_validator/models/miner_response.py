from __future__ import annotations
from dataclasses import dataclass

import bittensor as bt
import json

from constants import (
    DEFAULT_PROOF_SIZE,
    SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
    VALIDATOR_REQUEST_TIMEOUT_SECONDS,
)


@dataclass
class MinerResponse:
    """
    Represents a response from a miner.

    Attributes:
        uid (int): Unique identifier of the miner.
        verification_result (bool): Whether the miner's response was verified.
        response_time (float): Time taken by the miner to respond.
        proof_size (int): Size of the proof provided by the miner.
        model_id (str): Identifier of the model used.
        proof_json (Any): JSON representation of the proof.
        raw (str): Deserialized form of the response.
        error (str): Error message, if any occurred during processing.
    """

    uid: int
    verification_result: bool
    response_time: float
    proof_size: int
    model_id: str
    proof_json: dict | None = None
    public_json: list[str] | None = None
    raw: dict | None = None
    error: str | None = None

    @classmethod
    def from_raw_response(cls, response: dict) -> "MinerResponse":
        """
        Creates a MinerResponse object from a raw response dictionary.

        Args:
            response (dict): Raw response from a miner.

        Returns:
            MinerResponse: Processed miner response object.
        """
        try:
            if isinstance(response, str):
                response = json.loads(response)
            deserialized_response = response.get("deserialized")

            proof_json = None
            public_json = None
            if isinstance(deserialized_response, str):
                try:
                    deserialized_response = json.loads(deserialized_response)
                except json.JSONDecodeError as e:
                    bt.logging.debug(f"JSON decoding error: {e}")
                    return cls.empty(uid=response.get("uid", 0))

            if isinstance(deserialized_response, dict):
                proof = deserialized_response.get("proof", "{}")
                public_signals = deserialized_response.get("public_signals", "[]")

                if proof:
                    proof_json = json.loads(proof) if isinstance(proof, str) else proof
                if public_signals:
                    public_json = (
                        json.loads(public_signals)
                        if isinstance(public_signals, str)
                        else public_signals
                    )

            proof_size = (
                sum(
                    len(str(value))
                    for key in ("pi_a", "pi_b", "pi_c")
                    for element in proof_json.get(key, [])
                    for value in (element if isinstance(element, list) else [element])
                )
                if proof_json
                else DEFAULT_PROOF_SIZE
            )

            return cls(
                uid=response.get("uid", 0),
                verification_result=False,
                response_time=response.get(
                    "response_time", VALIDATOR_REQUEST_TIMEOUT_SECONDS
                ),
                proof_size=proof_size,
                model_id=response.get("model_id", SINGLE_PROOF_OF_WEIGHTS_MODEL_ID),
                proof_json=proof_json,
                public_json=public_json,
                raw=deserialized_response,
            )
        except json.JSONDecodeError as e:
            bt.logging.error(f"JSON decoding error: {e}")
            return cls.empty(uid=response.get("uid", 0))
        except Exception as e:
            bt.logging.error(f"Error processing miner response: {e}")
            return cls.empty(uid=response.get("uid", 0))

    @classmethod
    def empty(cls, uid: int = 0) -> "MinerResponse":
        """
        Creates an empty MinerResponse object.

        Returns:
            MinerResponse: An empty MinerResponse object.
        """
        return cls(
            uid=uid,
            verification_result=False,
            response_time=VALIDATOR_REQUEST_TIMEOUT_SECONDS,
            proof_size=DEFAULT_PROOF_SIZE,
            model_id=SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
            proof_json=None,
            public_json=None,
            raw=None,
            error="Empty response",
        )

    def set_verification_result(self, result: bool):
        """
        Sets the verification result for the miner's response.

        Args:
            result (bool): The verification result to set.
        """
        self.verification_result = result
