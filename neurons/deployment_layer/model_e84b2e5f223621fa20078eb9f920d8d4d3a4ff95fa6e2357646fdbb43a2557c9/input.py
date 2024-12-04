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
SCALING = 100000000


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
        if request_type == RequestType.RWR and data is not None:
            data = self._add_missing_constants(data)
        super().__init__(request_type, data)

    @staticmethod
    def generate() -> dict[str, object]:

        minimum_response_time = int(random.random() * 60 * SCALING)

        maximum_response_time = minimum_response_time + int(
            random.random() * 60 * SCALING
        )

        response_time = (
            int(
                random.random()
                * (maximum_response_time - minimum_response_time)
                * SCALING
            )
            + minimum_response_time
        )
        return {
            "maximum_score": [int(1.0 * SCALING) for _ in range(BATCH_SIZE)],
            "previous_score": [
                int(random.random() * SCALING) for _ in range(BATCH_SIZE)
            ],
            "verified": [random.choice([True, False]) for _ in range(BATCH_SIZE)],
            "proof_size": [
                int(random.randint(0, 10000) * SCALING) for _ in range(BATCH_SIZE)
            ],
            "validator_uid": [random.randint(0, 255) for _ in range(BATCH_SIZE)],
            "block_number": [
                random.randint(3000000, 10000000) for _ in range(BATCH_SIZE)
            ],
            "miner_uid": [random.randint(0, 255) for _ in range(BATCH_SIZE)],
            "minimum_response_time": [minimum_response_time for _ in range(BATCH_SIZE)],
            "maximum_response_time": [maximum_response_time for _ in range(BATCH_SIZE)],
            "response_time": [response_time for _ in range(BATCH_SIZE)],
            "scaling": SCALING,
            "RATE_OF_DECAY": int(RATE_OF_DECAY * SCALING),
            "RATE_OF_RECOVERY": int(RATE_OF_RECOVERY * SCALING),
            "FLATTENING_COEFFICIENT": int(FLATTENING_COEFFICIENT * SCALING),
            "PROOF_SIZE_WEIGHT": int(PROOF_SIZE_WEIGHT * SCALING),
            "PROOF_SIZE_THRESHOLD": int(PROOF_SIZE_THRESHOLD * SCALING),
            "RESPONSE_TIME_WEIGHT": int(RESPONSE_TIME_WEIGHT * SCALING),
            "MAXIMUM_RESPONSE_TIME_DECIMAL": int(
                MAXIMUM_RESPONSE_TIME_DECIMAL * SCALING
            ),
        }

    @staticmethod
    def validate(data: dict[str, object]) -> None:
        return CircuitInputSchema(**data)

    def _add_missing_constants(self, data: dict[str, object]) -> dict[str, object]:
        for i in range(16):
            data["validator_uid"][BATCH_SIZE - 16 + i] = secrets.randbits(16)

        constants = [
            "RATE_OF_DECAY",
            "RATE_OF_RECOVERY",
            "FLATTENING_COEFFICIENT",
            "PROOF_SIZE_WEIGHT",
            "PROOF_SIZE_THRESHOLD",
            "RESPONSE_TIME_WEIGHT",
            "MAXIMUM_RESPONSE_TIME_DECIMAL",
        ]

        for constant in constants:
            if constant not in data:
                data[constant] = int(globals()[constant] * SCALING)

        if "scaling" not in data:
            data["scaling"] = SCALING

        return data

    @staticmethod
    def process(data: dict[str, object]) -> dict[str, object]:

        data["maximum_score"] = int(data["maximum_score"] * SCALING)
        data["previous_score"] = int(data["previous_score"] * SCALING)
        data["proof_size"] = int(data["proof_size"] * SCALING)
        data["minimum_response_time"] = int(data["minimum_response_time"] * SCALING)
        data["maximum_response_time"] = int(data["maximum_response_time"] * SCALING)
        data["response_time"] = int(data["response_time"] * SCALING)

        return data
