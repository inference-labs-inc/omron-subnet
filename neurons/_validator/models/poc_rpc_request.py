from _validator.models.base_rpc_request import RealWorldRequest
from pydantic import Field
from deployment_layer.circuit_store import circuit_store


class ProofOfComputationRPCRequest(RealWorldRequest):
    """
    Request for the Proof of Computation RPC method.
    """

    circuit_id: str = Field(..., description="The ID of the circuit to use")

    def __init__(self, circuit_id: str, inputs: dict[str, any]):
        circuit = circuit_store.get_circuit(circuit_id)
        if circuit is None:
            raise ValueError(f"No circuit found for ID {circuit_id}")
        super().__init__(circuit=circuit, inputs=inputs)
