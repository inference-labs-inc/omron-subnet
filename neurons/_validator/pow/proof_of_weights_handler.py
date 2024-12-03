import typing
from bittensor import logging
from _validator.utils.proof_of_weights import ProofOfWeightsItem
from execution_layer.circuit import Circuit, CircuitType
from constants import (
    SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
)
from protocol import ProofOfWeightsSynapse, QueryZkProof

if typing.TYPE_CHECKING:
    from _validator.utils.request_type import RequestType


class ProofOfWeightsHandler:

    @staticmethod
    def prepare_pow_request(circuit: Circuit, proof_of_weights_queue):
        if not proof_of_weights_queue:
            logging.warning("No proof of weights queue found. Defaulting to benchmark.")
            return (
                ProofOfWeightsSynapse(
                    subnet_uid=circuit.metadata.netuid,
                    verification_key_hash=circuit.id,
                    proof_system=circuit.proof_system,
                    inputs=circuit.input_handler(RequestType.BENCHMARK).to_json(),
                    proof="",
                    public_signals="",
                )
                if circuit.metadata.type == CircuitType.PROOF_OF_WEIGHTS
                else QueryZkProof(
                    model_id=circuit.id,
                    query_input=circuit.input_handler(RequestType.BENCHMARK).to_json(),
                    query_output="",
                )
            )

        pow_items: list[ProofOfWeightsItem] = ProofOfWeightsItem.pad_items(
            proof_of_weights_queue,
            target_item_count=(
                256 if circuit.id == SINGLE_PROOF_OF_WEIGHTS_MODEL_ID else 1024
            ),
        )
        inputs = circuit.input_handler(
            RequestType.RWR, ProofOfWeightsItem.to_dict_list(pow_items)
        ).to_json()
        if circuit.metadata.type == CircuitType.PROOF_OF_WEIGHTS:
            return ProofOfWeightsSynapse(
                subnet_uid=circuit.metadata.netuid,
                verification_key_hash=circuit.id,
                proof_system=circuit.proof_system,
                inputs=inputs,
                proof="",
                public_signals="",
            )
        return QueryZkProof(
            model_id=circuit.id,
            query_input=inputs,
            query_output="",
        )
