from __future__ import annotations
from pydantic import BaseModel
from execution_layer.base_input import BaseInput
from execution_layer.input_registry import InputRegistry
from _validator.models.request_type import RequestType
import numpy as np

BATCH_SIZE = 1
SEQUENCE_LENGTH = 64


class CircuitInputSchema(BaseModel):
    input_ids: list[list[int]]
    shapes: list[list[int]]
    variables: list[list[object]]


@InputRegistry.register(
    "815d49886a610c17e34204b1c544c363bb06328a6cb126181bd0b6e77164cd4b"
)
class CircuitInput(BaseInput):

    schema = CircuitInputSchema

    def __init__(
        self, request_type: RequestType, data: dict[str, object] | None = None
    ):
        super().__init__(request_type, data)

    @staticmethod
    def generate() -> dict[str, object]:
        input_ids = np.random.randint(
            0, 256, size=(BATCH_SIZE, SEQUENCE_LENGTH)
        ).tolist()
        return {
            "input_ids": input_ids,
            "shapes": [[BATCH_SIZE, SEQUENCE_LENGTH]],
            "variables": [["batch_size", BATCH_SIZE], ["sequence", SEQUENCE_LENGTH]],
        }

    @staticmethod
    def validate(data: dict[str, object]) -> None:
        return CircuitInputSchema(**data)

    @staticmethod
    def process(data: dict[str, object]) -> dict[str, object]:
        return data
