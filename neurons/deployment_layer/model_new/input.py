from __future__ import annotations
from pydantic import BaseModel
from execution_layer.base_input import BaseInput
from execution_layer.input_registry import InputRegistry
from _validator.models.request_type import RequestType
import random

BATCH_SIZE = 1
CHANNELS = 3
HEIGHT = 416
WIDTH = 416
INPUT_SIZE = BATCH_SIZE * CHANNELS * HEIGHT * WIDTH


class CircuitInputSchema(BaseModel):
    inputs: list[float]


@InputRegistry.register("new")
class CircuitInput(BaseInput):
    schema = CircuitInputSchema

    def __init__(
        self, request_type: RequestType, data: dict[str, object] | None = None
    ):
        super().__init__(request_type, data)

    @staticmethod
    def generate() -> dict[str, object]:
        return {
            "inputs": [random.random() for _ in range(INPUT_SIZE)],
        }

    @staticmethod
    def validate(data: dict[str, object]) -> None:
        return CircuitInputSchema(**data)

    @staticmethod
    def process(data: dict[str, object]) -> dict[str, object]:
        return data
