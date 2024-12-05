from __future__ import annotations
from pydantic import BaseModel
from execution_layer.base_input import BaseInput
from execution_layer.input_registry import InputRegistry
from _validator.models.request_type import RequestType
import random
import secrets

BATCH_SIZE = 256


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
    uid_responsible_for_proof: int


@InputRegistry.register(
    "37320fc74fec80805eedc8e92baf3c58842a2cb2a4ae127ad6e930f0c8441c7a"
)
class CircuitInput(BaseInput):
    def __init__(
        self, request_type: RequestType, data: dict[str, object] | None = None
    ):
        super().__init__(request_type, data)

    @staticmethod
    def generate() -> dict[str, object]:
        data = {
            "maximum_score": [random.random() for _ in range(BATCH_SIZE)],
            "previous_score": [random.random() for _ in range(BATCH_SIZE)],
            "verified": [random.choice([True, False]) for _ in range(BATCH_SIZE)],
            "proof_size": [random.random() * 5000 for _ in range(BATCH_SIZE)],
            "validator_uid": [random.randint(0, 255) for _ in range(BATCH_SIZE)],
            "block_number": [random.randint(0, 100000) for _ in range(BATCH_SIZE)],
            "miner_uid": [random.randint(0, 255) for _ in range(BATCH_SIZE)],
            "uid_responsible_for_proof": random.randint(0, 255),
        }

        data["minimum_response_time"] = [
            random.random() * 60 for _ in range(BATCH_SIZE)
        ]
        data["maximum_response_time"] = [
            min_time + 1 + random.random() for min_time in data["minimum_response_time"]
        ]
        data["response_time"] = [
            min_time + random.random() * (max_time - min_time)
            for min_time, max_time in zip(
                data["minimum_response_time"], data["maximum_response_time"]
            )
        ]

        return data

    @staticmethod
    def validate(data: dict[str, object]) -> None:
        return CircuitInputSchema(**data)

    @staticmethod
    def process(data: dict[str, object]) -> dict[str, object]:
        for i in range(16):
            data["validator_uid"][BATCH_SIZE - 16 + i] = secrets.randbits(16)
        return data
