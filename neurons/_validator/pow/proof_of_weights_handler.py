import math
import numpy.random as rand
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
from execution_layer.circuit import Circuit
from constants import (
    SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
    BATCHED_PROOF_OF_WEIGHTS_MODEL_ID,
)
from execution_layer.circuit import ProofSystem
from protocol import ProofOfWeightsSynapse, QueryZkProof
from deployment_layer.circuit_store import circuit_store
import secrets


class ProofOfWeightsHandler:
    use_sn27 = False

    @staticmethod
    def prepare_pow_request(proof_of_weights_queue, subnet_uid):
        logging.debug(f"Preparing PoW request for subnet UID: {subnet_uid}")

        # Get appropriate circuit based on subnet
        circuit = ProofOfWeightsHandler._get_circuit_for_subnet(
            subnet_uid, proof_of_weights_queue
        )

        serialized_items = []
        inputs = []

        if subnet_uid in (2, 118):
            pow_items = ProofOfWeightsHandler._prepare_pow_items(
                proof_of_weights_queue, circuit
            )
            serialized_items = ProofOfWeightsItem.to_dict_list(pow_items)
            inputs = ProofOfWeightsHandler._prepare_inputs(serialized_items, circuit)
        else:
            serialized_items = (
                proof_of_weights_queue[0] if proof_of_weights_queue else []
            )
            inputs = serialized_items

        synapse = ProofOfWeightsHandler._create_synapse(subnet_uid, inputs, circuit)

        return {
            "synapse": synapse,
            "inputs": inputs,
            "model_id": circuit.id,
            "aggregation": False,
        }

    @staticmethod
    def _get_circuit_for_subnet(
        subnet_uid: int, proof_of_weights_queue: list
    ) -> Circuit:
        """Get the appropriate circuit based on subnet and queue state."""
        if subnet_uid not in (2, 118):
            return circuit_store.get_latest_circuit_for_netuid(subnet_uid)

        use_batched_pow, _ = ProofOfWeightsHandler._determine_pow_type(
            proof_of_weights_queue
        )

        if use_batched_pow:
            return circuit_store.get_circuit(BATCHED_PROOF_OF_WEIGHTS_MODEL_ID)

        if subnet_uid == 27:
            ProofOfWeightsHandler.use_sn27 = not ProofOfWeightsHandler.use_sn27
            return (
                circuit_store.get_latest_circuit_for_netuid(27)
                if ProofOfWeightsHandler.use_sn27
                else circuit_store.get_circuit(SINGLE_PROOF_OF_WEIGHTS_MODEL_ID)
            )

        return circuit_store.get_circuit(SINGLE_PROOF_OF_WEIGHTS_MODEL_ID)

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
    def _prepare_pow_items(proof_of_weights_queue, circuit) -> list[ProofOfWeightsItem]:
        """Prepare and pad the PoW items based on the determined PoW type."""
        logging.trace("Preparing PoW items")
        if circuit.id == BATCHED_PROOF_OF_WEIGHTS_MODEL_ID:
            pow_items = ProofOfWeightsItem.pad_items(
                proof_of_weights_queue, target_item_count=1024
            )
            logging.debug("Prepared batched PoW items")
        elif circuit.id == SINGLE_PROOF_OF_WEIGHTS_MODEL_ID:
            pow_items = ProofOfWeightsItem.pad_items(
                proof_of_weights_queue, target_item_count=256
            )
            logging.debug("Prepared single PoW items")
        else:
            pow_items = [ProofOfWeightsItem.empty()] * 256
            logging.debug("Prepared empty PoW items")

        return pow_items

    @staticmethod
    def _create_synapse(
        subnet_uid: int,
        inputs: dict,
        circuit: Circuit,
    ) -> ProofOfWeightsSynapse | QueryZkProof:
        """Create the appropriate synapse based on the PoW type."""
        logging.trace("Creating synapse")
        if circuit.id == BATCHED_PROOF_OF_WEIGHTS_MODEL_ID:
            synapse = ProofOfWeightsSynapse(
                subnet_uid=subnet_uid,
                verification_key_hash=circuit.id,
                proof_system=ProofSystem.CIRCOM,
                inputs=inputs,
                proof="",
                public_signals="",
            )
            logging.debug(f"Created batched SN2 PoW synapse with model: {circuit}")
        elif subnet_uid == 27:
            synapse = ProofOfWeightsSynapse(
                subnet_uid=subnet_uid,
                verification_key_hash=circuit.id,
                proof_system=circuit.proof_system,
                inputs=inputs,
                proof="",
                public_signals="",
            )
            logging.debug(f"Created batched PoW synapse for SN27 with model: {circuit}")
        else:
            synapse = QueryZkProof(
                query_input={
                    "model_id": circuit.id,
                    "public_inputs": inputs,
                }
            )
            logging.debug(f"Created single synapse for SN2 with model: {circuit}")
        return synapse

    @staticmethod
    def _prepare_inputs(
        serialized_items: dict,
        circuit: Circuit,
    ) -> dict:
        """Prepare the inputs for the Proof of Weights circuit."""
        scaling = circuit.settings.get("scaling", 100000000)

        if circuit.proof_system == ProofSystem.CIRCOM:
            return ProofOfWeightsHandler._prepare_circom_inputs(
                serialized_items, scaling
            )
        elif circuit.metadata.netuid == 27:
            return ProofOfWeightsHandler._prepare_subnet27_inputs(serialized_items)

        return ProofOfWeightsHandler._prepare_base_inputs(serialized_items, scaling)

    @staticmethod
    def _prepare_base_inputs(serialized_items: dict, scaling: int) -> dict:
        """Prepare base inputs common to all circuits."""
        return {
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
        }

    @staticmethod
    def _prepare_circom_inputs(serialized_items: dict, scaling: int) -> dict:
        """Prepare inputs for CIRCOM circuits."""
        inputs = {
            "scaling": scaling,
            "RATE_OF_DECAY": int(RATE_OF_DECAY * scaling),
            "RATE_OF_RECOVERY": int(RATE_OF_RECOVERY * scaling),
            "FLATTENING_COEFFICIENT": int(FLATTENING_COEFFICIENT * scaling),
            "PROOF_SIZE_WEIGHT": int(PROOF_SIZE_WEIGHT * scaling),
            "PROOF_SIZE_THRESHOLD": int(PROOF_SIZE_THRESHOLD * scaling),
            "RESPONSE_TIME_WEIGHT": int(RESPONSE_TIME_WEIGHT * scaling),
            "MAXIMUM_RESPONSE_TIME_DECIMAL": int(
                MAXIMUM_RESPONSE_TIME_DECIMAL * scaling
            ),
        }
        return inputs

    @staticmethod
    def _prepare_subnet48_inputs(serialized_items: dict) -> dict:
        """Prepare inputs for SN48 circuits."""
        inputs = {
            "scores": [rand.random() for _ in range(256)],
            "top_tier_pct": 0.2,
            "next_tier_pct": 0.3,
            "top_tier_weight": 1.0,
            "next_tier_weight": 0.5,
            "bottom_tier_weight": 0.1,
            "validator_uid": serialized_items["validator_uid"],
            "nonce": secrets.randbits(32),
        }
        return inputs

    @staticmethod
    def _prepare_subnet27_inputs(serialized_items: dict) -> dict:
        """Prepare inputs for SN27 circuits."""

        SUCCESS_WEIGHT = 1
        DIFFICULTY_WEIGHT = 1
        TIME_ELAPSED_WEIGHT = 0.3
        FAILED_PENALTY_WEIGHT = 0.4
        ALLOCATION_WEIGHT = 0.21
        POW_TIMEOUT = 30
        POW_MIN_DIFFICULTY = 7
        POW_MAX_DIFFICULTY = 12

        inputs = {
            "challenge_attempts": serialized_items.get("challenge_attempts", 1),
            "challenge_successes": serialized_items.get("challenge_successes", 0),
            "last_20_challenge_failed": serialized_items.get(
                "last_20_challenge_failed", 0
            ),
            "challenge_elapsed_time_avg": serialized_items.get(
                "challenge_elapsed_time_avg", POW_TIMEOUT
            ),
            "last_20_difficulty_avg": serialized_items.get(
                "last_20_difficulty_avg", POW_MIN_DIFFICULTY
            ),
            "has_docker": serialized_items.get("has_docker", False),
            "uid": serialized_items["validator_uid"],
            "allocated_uids": serialized_items.get("allocated_uids", []),
            "penalized_uids": serialized_items.get("penalized_uids", []),
            "validator_uids": serialized_items.get("validator_uids", []),
            "success_weight": SUCCESS_WEIGHT,
            "difficulty_weight": DIFFICULTY_WEIGHT,
            "time_elapsed_weight": TIME_ELAPSED_WEIGHT,
            "failed_penalty_weight": FAILED_PENALTY_WEIGHT,
            "allocation_weight": ALLOCATION_WEIGHT,
            "pow_timeout": POW_TIMEOUT,
            "pow_min_difficulty": POW_MIN_DIFFICULTY,
            "pow_max_difficulty": POW_MAX_DIFFICULTY,
            "nonce": secrets.randbits(32),
        }
        return inputs
