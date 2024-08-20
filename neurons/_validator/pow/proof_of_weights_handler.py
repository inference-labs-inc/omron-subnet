import math
import torch
from bittensor import logging
from _validator.utils.proof_of_weights import ProofOfWeightsItem
from _validator.scoring.reward import (
    FLATTENING_COEFFICIENT,
    MAXIMUM_RESPONSE_TIME_DECIMAL,
    PROOF_SIZE_THRESHOLD,
    PROOF_SIZE_WEIGHT,
    RATE_OF_DECAY,
    RATE_OF_RECOVERY,
    RESPONSE_TIME_WEIGHT,
)
from constants import (
    SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
    BATCHED_PROOF_OF_WEIGHTS_MODEL_ID,
)
from execution_layer.circuit import ProofSystem
from protocol import ProofOfWeightsSynapse, QueryZkProof
from deployment_layer.circuit_store import circuit_store


class ProofOfWeightsHandler:
    @staticmethod
    def prepare_pow_request(uid, proof_of_weights_queue, subnet_uid):
        logging.debug(
            f"Preparing PoW request for validator UID: {uid}, subnet UID: {subnet_uid}"
        )
        use_batched_pow, use_single_pow = ProofOfWeightsHandler._determine_pow_type(
            proof_of_weights_queue
        )

        pow_circuit = ProofOfWeightsHandler._get_pow_circuit()
        scaling = pow_circuit.settings["scaling"]
        logging.trace(f"PoW circuit scaling: {scaling}")

        pow_items = ProofOfWeightsHandler._prepare_pow_items(
            proof_of_weights_queue, use_batched_pow, use_single_pow, uid
        )

        serialized_items = ProofOfWeightsItem.to_dict_list(pow_items)

        inputs = ProofOfWeightsHandler._prepare_inputs(serialized_items, scaling)

        synapse, model_id = ProofOfWeightsHandler._create_synapse(
            use_batched_pow, subnet_uid, inputs
        )

        logging.debug(f"PoW request prepared with model ID: {model_id}")
        return {
            "synapse": synapse,
            "inputs": inputs,
            "model_id": model_id,
            "aggregation": False,
        }

    @staticmethod
    def _determine_pow_type(proof_of_weights_queue):
        """Determine whether to use batched or single PoW based on queue state."""
        queue_length = len(proof_of_weights_queue)
        logging.trace(f"Determining PoW type. Queue length: {queue_length}")
        if queue_length > 0:
            use_batched_pow = queue_length >= 1024
            use_single_pow = queue_length > 0 and not use_batched_pow
        else:
            use_batched_pow = use_single_pow = False
        logging.debug(
            f"PoW type determined: Batched: {use_batched_pow}, Single: {use_single_pow}"
        )
        return use_batched_pow, use_single_pow

    @staticmethod
    def _get_pow_circuit():
        """Retrieve the PoW circuit from the circuit store."""
        logging.trace("Retrieving PoW circuit from circuit store")
        pow_circuit = circuit_store.get_circuit(SINGLE_PROOF_OF_WEIGHTS_MODEL_ID)
        if not pow_circuit:
            logging.error(
                f"PoW circuit not found for model ID: {SINGLE_PROOF_OF_WEIGHTS_MODEL_ID}"
            )
            raise ValueError(
                f"Proof of weights circuit not found for model ID: {SINGLE_PROOF_OF_WEIGHTS_MODEL_ID}"
            )
        logging.debug("PoW circuit retrieved successfully")
        return pow_circuit

    @staticmethod
    def _prepare_pow_items(
        proof_of_weights_queue, use_batched_pow, use_single_pow, uid
    ):
        """Prepare and pad the PoW items based on the determined PoW type."""
        logging.trace("Preparing PoW items")
        if use_batched_pow:
            pow_items = ProofOfWeightsItem.pad_items(
                proof_of_weights_queue, target_item_count=1024
            )
            logging.debug("Prepared batched PoW items")
        elif use_single_pow:
            pow_items = ProofOfWeightsItem.pad_items(
                proof_of_weights_queue, target_item_count=256
            )
            logging.debug("Prepared single PoW items")
        else:
            pow_items = [ProofOfWeightsItem.empty()] * 256
            logging.debug("Prepared empty PoW items")

        pow_items[-1].validator_uid = torch.tensor(uid, dtype=torch.int64)
        logging.trace(f"Set validator UID {uid} for last PoW item")
        return pow_items

    @staticmethod
    def _create_synapse(use_batched_pow, subnet_uid, inputs):
        """Create the appropriate synapse based on the PoW type."""
        logging.trace("Creating synapse")
        if use_batched_pow:
            model_id = BATCHED_PROOF_OF_WEIGHTS_MODEL_ID
            synapse = ProofOfWeightsSynapse(
                subnet_uid=subnet_uid,
                verification_key_hash=model_id,
                proof_system=ProofSystem.CIRCOM,
                inputs=inputs,
                proof="",
                public_signals="",
            )
            logging.debug(f"Created batched PoW synapse with model ID: {model_id}")
        else:
            model_id = SINGLE_PROOF_OF_WEIGHTS_MODEL_ID
            synapse = QueryZkProof(
                query_input={
                    "model_id": model_id,
                    "public_inputs": inputs,
                }
            )
            logging.debug(f"Created single PoW synapse with model ID: {model_id}")
        return synapse, model_id

    @staticmethod
    def _prepare_inputs(serialized_items: dict, scaling: int) -> dict:
        """Prepare the inputs for the Proof of Weights circuit."""
        logging.trace("Preparing inputs for PoW circuit")
        inputs = {
            "RATE_OF_DECAY": int(RATE_OF_DECAY * scaling),
            "RATE_OF_RECOVERY": int(RATE_OF_RECOVERY * scaling),
            "FLATTENING_COEFFICIENT": int(FLATTENING_COEFFICIENT * scaling),
            "PROOF_SIZE_WEIGHT": int(PROOF_SIZE_WEIGHT * scaling),
            "PROOF_SIZE_THRESHOLD": int(PROOF_SIZE_THRESHOLD * scaling),
            "RESPONSE_TIME_WEIGHT": int(RESPONSE_TIME_WEIGHT * scaling),
            "MAXIMUM_RESPONSE_TIME_DECIMAL": int(
                MAXIMUM_RESPONSE_TIME_DECIMAL * scaling
            ),
            "maximum_score": [
                int(score * scaling) for score in serialized_items["max_score"]
            ],
            "previous_score": [
                int(score if not math.isnan(score) else 0)
                for score in serialized_items["previous_score"]
            ],
            "verified": serialized_items["verification_result"],
            "proof_size": [
                int(size * scaling) for size in serialized_items["proof_size"]
            ],
            "response_time": [
                int(time * scaling) for time in serialized_items["response_time"]
            ],
            "maximum_response_time": [
                int(time * scaling)
                for time in serialized_items["median_max_response_time"]
            ],
            "minimum_response_time": [
                int(time * scaling) for time in serialized_items["min_response_time"]
            ],
            "block_number": serialized_items["block_number"],
            "validator_uid": serialized_items["validator_uid"],
            "miner_uid": serialized_items["uid"],
            "scaling": scaling,
        }
        logging.debug("Inputs prepared for PoW circuit")
        return inputs
