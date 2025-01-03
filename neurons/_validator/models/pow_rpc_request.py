from __future__ import annotations
from _validator.models.base_rpc_request import RealWorldRequest
from pydantic import Field
from deployment_layer.circuit_store import circuit_store


class ProofOfWeightsRPCRequest(RealWorldRequest):
    """
    Request for the Proof of Weights RPC method.
    """

    weights_version: int | None = Field(
        ..., description="The version of weights in use by the origin subnet"
    )
    netuid: int = Field(..., description="The origin subnet UID")

    def __init__(
        self, weights_version: int, evaluation_data: dict[str, any], netuid: int
    ):
        circuit = circuit_store.get_circuit_for_netuid_and_version(
            netuid=netuid, version=weights_version
        )
        if circuit is None:
            raise ValueError(
                f"No circuit found for netuid {netuid} and weights version {weights_version}"
            )
        super().__init__(circuit=circuit, inputs=evaluation_data)
