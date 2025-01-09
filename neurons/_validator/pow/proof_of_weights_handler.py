from bittensor import logging
from _validator.utils.proof_of_weights import ProofOfWeightsItem
from execution_layer.circuit import Circuit, CircuitType
from constants import (
    SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
)
from protocol import ProofOfWeightsSynapse, QueryZkProof
from _validator.models.request_type import RequestType


class ProofOfWeightsHandler:
    """
    Handles internal proof of weights
    This covers the case where the origin validator is a validator on Omron;
    no external requests are needed as this internal mechanism is used to generate the proof of weights.
    """

    @staticmethod
    def prepare_pow_request(circuit: Circuit, proof_of_weights_queue):
        if not proof_of_weights_queue:
            logging.debug("No proof of weights queue found. Defaulting to benchmark.")
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
                    query_input={
                        "public_inputs": circuit.input_handler(
                            RequestType.BENCHMARK
                        ).to_json(),
                        "model_id": circuit.id,
                    },
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
            query_input={"public_inputs": inputs, "model_id": circuit.id},
            query_output="",
        )
