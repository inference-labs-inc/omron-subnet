from __future__ import annotations
from _validator.models.base_rpc_request import RealWorldRequest
from pydantic import Field
from deployment_layer.circuit_store import circuit_store
from execution_layer.generic_input import GenericInput
from _validator.models.request_type import RequestType


class ProofOfWeightsRPCRequest(RealWorldRequest):
    """
    Request for the Proof of Weights RPC method.
    """

    weights_version: int | None = Field(
        None, description="The version of weights in use by the origin subnet"
    )
    netuid: int = Field(..., description="The origin subnet UID")
    evaluation_data: dict[str, any] = Field(
        ..., description="The evaluation data for the request"
    )

    def __init__(self, **data):
        netuid = data.get("netuid")
        weights_version = data.get("weights_version")
        evaluation_data = data.get("evaluation_data")

        circuit = None
        if weights_version is None:
            circuit = circuit_store.get_latest_circuit_for_netuid(netuid)
            weights_version = circuit.metadata.weights_version
        else:
            circuit = circuit_store.get_circuit_for_netuid_and_version(
                netuid=netuid, version=weights_version
            )
        if circuit is None:
            raise ValueError(
                f"No circuit found for netuid {netuid} and weights version {weights_version}"
            )

        super().__init__(
            circuit=circuit,
            inputs=GenericInput(RequestType.RWR, evaluation_data),
            **data,
        )
