from __future__ import annotations
from pydantic import BaseModel
from execution_layer.base_input import BaseInput
from execution_layer.input_registry import InputRegistry
from _validator.models.request_type import RequestType
import random

LIST_SIZE = 5


class CircuitInputSchema(BaseModel):
    list_items: list[float]


@InputRegistry.register(
    "50818a54b31b3e0fe3306a7fb7826156fc2c42c9d64c6ba106ba135fbe7b7b19"
)
class CircuitInput(BaseInput):
    def __init__(
        self, request_type: RequestType, data: dict[str, object] | None = None
    ):
        super().__init__(request_type, data)

    @staticmethod
    def generate() -> dict[str, object]:
        return {
            "list_items": [random.uniform(0.0, 0.85) for _ in range(LIST_SIZE)],
        }

    @staticmethod
    def validate(data: dict[str, object]) -> None:
        return CircuitInputSchema(**data)

    @staticmethod
    def process(data: dict[str, object]) -> dict[str, object]:
        """
        No processing needs to take place, as all inputs are randomized.
        """
        return data
