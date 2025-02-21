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
    "96a1315ab963a5a906fb9c7515f60ba386d58310c0ad47bb1049f2dc9b2bdd64"
)
class CircuitInput(BaseInput):

    schema = CircuitInputSchema

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
