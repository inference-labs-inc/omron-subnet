from dataclasses import dataclass

from execution_layer.circuit import Circuit
from _validator.models.request_type import RequestType
from protocol import QueryZkProof, ProofOfWeightsSynapse
from execution_layer.generic_input import GenericInput
import bittensor as bt


@dataclass
class Request:
    uid: int
    axon: bt.axon
    synapse: QueryZkProof | ProofOfWeightsSynapse
    circuit: Circuit
    request_type: RequestType
    inputs: GenericInput | None = None
    response_time: float | None = None
    deserialized: dict[str, object] | None = None
    result: bt.Synapse | None = None
