from pydantic import BaseModel, Field
from execution_layer.circuit import Circuit
from _validator.utils.api import hash_inputs


class RealWorldRequest(BaseModel):
    """
    Base class for all incoming RPC requests.
    """

    circuit: Circuit = Field(
        ..., description="The circuit selected to handle the request"
    )
    inputs: dict[str, any] = Field(..., description="The inputs to the circuit")
    hash: str = Field(..., description="The hash of request inputs")

    def __init__(self, circuit: Circuit, inputs: dict[str, any]):
        super().__init__(circuit=circuit, inputs=inputs)
        self.hash = hash_inputs(inputs)
