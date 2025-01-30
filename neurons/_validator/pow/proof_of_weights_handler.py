from bittensor import logging
from _validator.utils.proof_of_weights import ProofOfWeightsItem
from execution_layer.circuit import Circuit, CircuitType
from constants import (
    BATCHED_PROOF_OF_WEIGHTS_MODEL_ID,
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
    def prepare_pow_request(
        circuit: Circuit, score_manager
    ) -> ProofOfWeightsSynapse | QueryZkProof:
        queue = score_manager.get_pow_queue()
        batch_size = 1024

        if circuit.id != BATCHED_PROOF_OF_WEIGHTS_MODEL_ID:
            logging.debug("Not a batched PoW model. Defaulting to benchmark.")
            return None, False

        if len(queue) < batch_size:
            logging.debug(
                f"Queue is less than {batch_size} items. Defaulting to benchmark."
            )
            return None, False

        pow_items = ProofOfWeightsItem.pad_items(
            queue[:batch_size], target_item_count=batch_size
        )

        logging.info(f"Preparing PoW request for {str(circuit)}")
        score_manager.remove_processed_items(batch_size)
        return (
            ProofOfWeightsHandler._create_request_from_items(circuit, pow_items),
            True,
        )

    @staticmethod
    def _create_request_from_items(
        circuit: Circuit, pow_items: list[ProofOfWeightsItem]
    ) -> ProofOfWeightsSynapse | QueryZkProof:
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
