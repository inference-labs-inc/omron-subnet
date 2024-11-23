from typing import Any

from pydantic import BaseModel


class PowInputModel(BaseModel):
    """
    Pydantic model for incoming requests.
        - inputs: the proof of weights request - usially a list or dict.
        - netuid: The originating subnet the request comes from.
    """

    inputs: Any
    netuid: int
