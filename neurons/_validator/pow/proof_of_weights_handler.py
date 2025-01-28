from bittensor import logging
from _validator.utils.proof_of_weights import ProofOfWeightsItem
from execution_layer.circuit import Circuit, CircuitType
from constants import (
    SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
    BATCHED_PROOF_OF_WEIGHTS_MODEL_ID,
    VALIDATOR_REQUEST_TIMEOUT_SECONDS,
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
    def prepare_pow_request(circuit: Circuit, score_manager):
        queue = score_manager.get_pow_queue()

        if len(queue) == 0 or len(queue) % 256 != 0:
            logging.debug(
                "Queue is empty or not a multiple of 256. Defaulting to benchmark."
            )
            return ProofOfWeightsHandler._create_benchmark_request(circuit)

        # Try to process queue first
        if score_manager.process_pow_queue(circuit.id):
            logging.info("Queue processed successfully")
            # Get fresh queue after processing
            queue = score_manager.get_pow_queue()

        batch_size = 256 if circuit.id == SINGLE_PROOF_OF_WEIGHTS_MODEL_ID else 1024
        pow_items = ProofOfWeightsItem.pad_items(
            queue[:batch_size], target_item_count=batch_size
        )

        logging.info(f"Preparing PoW request with {len(queue)} items in queue")
        if circuit.id == BATCHED_PROOF_OF_WEIGHTS_MODEL_ID:
            score_manager.clear_pow_queue()
        return ProofOfWeightsHandler._create_request_from_items(circuit, pow_items)

    @staticmethod
    def _create_benchmark_request(circuit: Circuit):
        """Create a benchmark request when queue is empty."""
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

    @staticmethod
    def _create_request_from_items(
        circuit: Circuit, pow_items: list[ProofOfWeightsItem]
    ):
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
            if item.minimum_response_time >= item.maximum_response_time:
                logging.debug(
                    "Minimum response time is gte than maximum response time for item. Setting to default timeout."
                )
                item.minimum_response_time = torch.tensor(0, dtype=torch.float32)
                item.maximum_response_time = torch.tensor(
                    VALIDATOR_REQUEST_TIMEOUT_SECONDS, dtype=torch.float32
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
