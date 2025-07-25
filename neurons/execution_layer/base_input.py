from __future__ import annotations
from _validator.models.request_type import RequestType


class BaseInput:
    """Base class for circuit-specific input handlers"""

    schema = None

    def __init__(
        self, request_type: RequestType, data: dict[str, object] | None = None
    ):
        if request_type == RequestType.BENCHMARK:
            self.data = self.generate()
        else:
            self.data = self.process(data) if data else {}
        self.validate(self.data)

    @staticmethod
    def generate() -> dict[str, object]:
        return {}

    def to_json(self) -> dict:
        return self.data

    def to_dict(self) -> dict:
        return self.data

    @staticmethod
    def validate(data: dict[str, object]) -> None:
        pass

    @staticmethod
    def process(data: dict[str, object]) -> dict[str, object]:
        return data
