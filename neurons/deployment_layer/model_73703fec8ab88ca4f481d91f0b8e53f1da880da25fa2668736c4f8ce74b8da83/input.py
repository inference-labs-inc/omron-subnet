from __future__ import annotations
from pydantic import BaseModel
from execution_layer.base_input import BaseInput
from execution_layer.input_registry import InputRegistry
from _validator.models.request_type import RequestType
import random


class CircuitInputSchema(BaseModel):
    image: list[float]


@InputRegistry.register(
    "73703fec8ab88ca4f481d91f0b8e53f1da880da25fa2668736c4f8ce74b8da83"
)
class CircuitInput(BaseInput):
    def __init__(
        self, request_type: RequestType, data: dict[str, object] | None = None
    ):
        super().__init__(request_type, data)

    @staticmethod
    def generate() -> dict[str, object]:
        image = [random.random() for _ in range(224 * 224 * 3)]
        return {"image": image}

    @staticmethod
    def validate(data: dict[str, object]) -> None:
        return CircuitInputSchema(**data)

    @staticmethod
    def process(data: dict[str, object]) -> dict[str, object]:

        return data
