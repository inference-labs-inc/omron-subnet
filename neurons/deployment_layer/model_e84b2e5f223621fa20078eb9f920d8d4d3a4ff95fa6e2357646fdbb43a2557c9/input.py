from __future__ import annotations
from pydantic import BaseModel
from execution_layer.base_input import BaseInput
from execution_layer.input_registry import InputRegistry
from _validator.models.request_type import RequestType
import random
import secrets

BATCH_SIZE = 1024
RATE_OF_DECAY = 0.4
RATE_OF_RECOVERY = 0.1
FLATTENING_COEFFICIENT = 0.9
PROOF_SIZE_THRESHOLD = 3648
PROOF_SIZE_WEIGHT = 0
RESPONSE_TIME_WEIGHT = 1
MAXIMUM_RESPONSE_TIME_DECIMAL = 0.99


class CircuitInputSchema(BaseModel):
    maximum_score: list[float]
    previous_score: list[float]
    verified: list[bool]
    proof_size: list[float]
    response_time: list[float]
    maximum_response_time: list[float]
    minimum_response_time: list[float]
    validator_uid: list[int]
    block_number: list[int]
    miner_uid: list[int]
    scaling: int
    RATE_OF_DECAY: int
    RATE_OF_RECOVERY: int
    FLATTENING_COEFFICIENT: int
    PROOF_SIZE_WEIGHT: int
    PROOF_SIZE_THRESHOLD: int
    RESPONSE_TIME_WEIGHT: int
    MAXIMUM_RESPONSE_TIME_DECIMAL: int


@InputRegistry.register(
    "e84b2e5f223621fa20078eb9f920d8d4d3a4ff95fa6e2357646fdbb43a2557c9"
)
class CircuitInput(BaseInput):
    def __init__(
        self, request_type: RequestType, data: dict[str, object] | None = None
    ):
        super().__init__(request_type, data)

    @staticmethod
    def generate() -> dict[str, object]:
        scale = 1
        return {
            "maximum_score": [1.0 for _ in range(BATCH_SIZE)],
            "previous_score": [random.random() for _ in range(BATCH_SIZE)],
            "verified": [random.choice([True, False]) for _ in range(BATCH_SIZE)],
            "proof_size": [random.randint(0, 10000) for _ in range(BATCH_SIZE)],
            "validator_uid": [random.randint(0, 255) for _ in range(BATCH_SIZE)],
            "block_number": [
                random.randint(3000000, 10000000) for _ in range(BATCH_SIZE)
            ],
            "miner_uid": [random.randint(0, 255) for _ in range(BATCH_SIZE)],
            "minimum_response_time": [random.random() * 60 for _ in range(BATCH_SIZE)],
            "maximum_response_time": [60.0 for _ in range(BATCH_SIZE)],
            "response_time": [random.random() * 60 for _ in range(BATCH_SIZE)],
            "scaling": scale,
            "RATE_OF_DECAY": int(RATE_OF_DECAY * scale),
            "RATE_OF_RECOVERY": int(RATE_OF_RECOVERY * scale),
            "FLATTENING_COEFFICIENT": int(FLATTENING_COEFFICIENT * scale),
            "PROOF_SIZE_WEIGHT": int(PROOF_SIZE_WEIGHT * scale),
            "PROOF_SIZE_THRESHOLD": int(PROOF_SIZE_THRESHOLD * scale),
            "RESPONSE_TIME_WEIGHT": int(RESPONSE_TIME_WEIGHT * scale),
            "MAXIMUM_RESPONSE_TIME_DECIMAL": int(MAXIMUM_RESPONSE_TIME_DECIMAL * scale),
        }

    @staticmethod
    def validate(data: dict[str, object]) -> None:
        return CircuitInputSchema(**data)

    @staticmethod
    def process(data: dict[str, object]) -> dict[str, object]:
        for i in range(16):
            data["validator_uid"][BATCH_SIZE - 16 + i] = secrets.randbits(16)
        return data
