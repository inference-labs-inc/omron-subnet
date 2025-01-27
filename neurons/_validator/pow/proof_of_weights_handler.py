from bittensor import logging
from _validator.utils.proof_of_weights import ProofOfWeightsItem
from execution_layer.circuit import Circuit, CircuitType
from constants import (
    SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
)
from protocol import ProofOfWeightsSynapse, QueryZkProof
from _validator.models.request_type import RequestType
import torch


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

        batch_size = 256 if circuit.id == SINGLE_PROOF_OF_WEIGHTS_MODEL_ID else 1024
        pow_items: list[ProofOfWeightsItem] = ProofOfWeightsItem.pad_items(
            proof_of_weights_queue,
            target_item_count=batch_size,
        )

        # Update response times from circuit evaluation data
        for item in pow_items:
            if item.response_time < circuit.evaluation_data.minimum_response_time:
                item.response_time = torch.tensor(
                    circuit.evaluation_data.minimum_response_time, dtype=torch.float32
                )
            item.minimum_response_time = torch.tensor(
                circuit.evaluation_data.minimum_response_time, dtype=torch.float32
            )
            item.maximum_response_time = torch.tensor(
                circuit.evaluation_data.maximum_response_time, dtype=torch.float32
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
