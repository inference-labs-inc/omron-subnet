from pydantic import BaseModel
from execution_layer.circuit import Circuit
from execution_layer.generic_input import GenericInput
from _validator.utils.api import hash_inputs


class RealWorldRequest(BaseModel):
    circuit: Circuit
    inputs: GenericInput

    model_config = {"arbitrary_types_allowed": True}

    @property
    def hash(self) -> str:
        return hash_inputs(self.inputs)
