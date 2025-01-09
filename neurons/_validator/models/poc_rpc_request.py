from _validator.models.base_rpc_request import RealWorldRequest
from pydantic import Field
from deployment_layer.circuit_store import circuit_store


class ProofOfComputationRPCRequest(RealWorldRequest):
    """
    Request for the Proof of Computation RPC method.
    """

    circuit_name: str = Field(..., description="The name of the circuit to use")
    circuit_version: int | None = Field(
        ..., description="The version of the circuit to use"
    )

    def __init__(
        self, circuit_name: str, circuit_version: int | None, inputs: dict[str, any]
    ):
        if circuit_version is None:
            circuit = circuit_store.get_latest_circuit_by_name(circuit_name)
        else:
            circuit = circuit_store.get_circuit_by_name_and_version(
                circuit_name=circuit_name, circuit_version=circuit_version
            )
        if circuit is None:
            raise ValueError(
                f"No circuit found for name {circuit_name} and version {circuit_version}"
            )
        super().__init__(circuit=circuit, inputs=inputs)
