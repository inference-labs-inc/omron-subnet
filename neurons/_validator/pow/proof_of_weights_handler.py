import math
import numpy as np
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
    SINGLE_PROOF_OF_WEIGHTS_MODEL_ID_JOLT,
)
from execution_layer.circuit import ProofSystem
from protocol import ProofOfWeightsSynapse, QueryZkProof
from deployment_layer.circuit_store import circuit_store
import secrets


class ProofOfWeightsHandler:
    jolt_counter = 0
    use_sn27 = False

    @staticmethod
    def prepare_pow_request(proof_of_weights_queue, subnet_uid, is_localnet=False):
        logging.debug(f"Preparing PoW request for subnet UID: {subnet_uid}")
        model_id = SINGLE_PROOF_OF_WEIGHTS_MODEL_ID
        circuit = circuit_store.get_circuit(model_id)
        if subnet_uid in (2, 118) or is_localnet:
            use_batched_pow, use_single_pow = ProofOfWeightsHandler._determine_pow_type(
                proof_of_weights_queue
            )

            proof_system = (
                ProofSystem.JOLT
                if use_single_pow and ProofOfWeightsHandler.jolt_counter % 10 == 0
                else ProofSystem.CIRCOM
            )

            if use_single_pow:
                ProofOfWeightsHandler.jolt_counter += 1

            if use_batched_pow:
                model_id = BATCHED_PROOF_OF_WEIGHTS_MODEL_ID
            elif proof_system == ProofSystem.CIRCOM:
                model_id = SINGLE_PROOF_OF_WEIGHTS_MODEL_ID
            else:
                model_id = (
                    circuit_store.get_latest_circuit_for_netuid(27).id
                    if ProofOfWeightsHandler.use_sn27
                    else SINGLE_PROOF_OF_WEIGHTS_MODEL_ID_JOLT
                )
                ProofOfWeightsHandler.use_sn27 = not ProofOfWeightsHandler.use_sn27

            if model_id is None:
                raise ValueError(f"No circuit found for subnet_uid: {subnet_uid}")

            circuit = circuit_store.get_circuit(model_id)
        else:
            circuit = circuit_store.get_latest_circuit_for_netuid(subnet_uid)

        if circuit is None:
            raise ValueError(f"No circuit found for subnet_uid: {subnet_uid}")

        logging.debug(f"PoW circuit: {circuit}")
        logging.info(f"Preparing for requests with {circuit}")

        serialized_items = (
            proof_of_weights_queue[0] if len(proof_of_weights_queue) else []
        )
        inputs = serialized_items

        if subnet_uid in (2, 118) or is_localnet:
            pow_items: list[ProofOfWeightsItem] = (
                ProofOfWeightsHandler._prepare_pow_items(
                    proof_of_weights_queue, circuit
                )
            )
            serialized_items = ProofOfWeightsItem.to_dict_list(pow_items)
            inputs = ProofOfWeightsHandler._prepare_inputs(serialized_items, circuit)

        synapse = ProofOfWeightsHandler._create_synapse(subnet_uid, inputs, circuit)

        logging.debug(f"PoW request prepared with model ID: {circuit.id}")
        return {
            "synapse": synapse,
            "inputs": inputs,
            "model_id": circuit.id,
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
        logging.trace(f"Preparing inputs for {circuit}")
        scaling = circuit.settings.get("scaling", 100000000)
        inputs = {
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
        if circuit.proof_system == ProofSystem.CIRCOM:
            inputs["scaling"] = scaling
            inputs["RATE_OF_DECAY"] = int(RATE_OF_DECAY * scaling)
            inputs["RATE_OF_RECOVERY"] = int(RATE_OF_RECOVERY * scaling)
            inputs["FLATTENING_COEFFICIENT"] = int(FLATTENING_COEFFICIENT * scaling)
            inputs["PROOF_SIZE_WEIGHT"] = int(PROOF_SIZE_WEIGHT * scaling)
            inputs["PROOF_SIZE_THRESHOLD"] = int(PROOF_SIZE_THRESHOLD * scaling)
            inputs["RESPONSE_TIME_WEIGHT"] = int(RESPONSE_TIME_WEIGHT * scaling)
            inputs["MAXIMUM_RESPONSE_TIME_DECIMAL"] = int(
                MAXIMUM_RESPONSE_TIME_DECIMAL * scaling
            )
        if circuit.proof_system == ProofSystem.JOLT:
            inputs["uid_responsible_for_proof"] = inputs["validator_uid"][-1]

        if circuit.metadata.netuid == 27:
            inputs = {
                "success_weight": rand.uniform(0.8, 1.2),
                "difficulty_weight": rand.uniform(0.8, 1.2),
                "time_elapsed_weight": rand.uniform(0.2, 0.4),
                "failed_penalty_weight": rand.uniform(0.3, 0.5),
                "allocation_weight": rand.uniform(0.15, 0.25),
                "pow_min_difficulty": rand.randint(6, 8),
                "pow_max_difficulty": rand.randint(11, 13),
                "pow_timeout": rand.uniform(25, 35),
                "max_score_challenge": rand.uniform(200.0, 300.0),
                "max_score_allocation": rand.uniform(15.0, 25.0),
                "max_score": rand.uniform(220.0, 320.0),
                "failed_penalty_exp": rand.uniform(1.3, 1.7),
                "validator_uid": serialized_items["validator_uid"],
                "challenge_attempts": rand.randint(10, 10000, 256).tolist(),
                "last_20_challenge_failed": rand.randint(0, 11, 256).tolist(),
                "challenge_elapsed_time_avg": rand.uniform(0.001, 31, 256).tolist(),
                "challenge_difficulty_avg": rand.uniform(7, 12, 256).tolist(),
                "has_docker": rand.choice([True, False], 256).tolist(),
                "allocated_hotkey": rand.choice([True, False], 256).tolist(),
                "penalized_hotkey_count": rand.randint(0, 10, 256).tolist(),
                "half_validators": int(rand.randint(1, 10)),
                "nonce": secrets.randbits(128),
            }
            inputs["challenge_successes"] = rand.randint(
                np.array(inputs["challenge_attempts"]) // 2,
                inputs["challenge_attempts"],
            ).tolist()
            inputs["uid_responsible_for_proof"] = inputs["validator_uid"][-1]

        logging.debug(f"Inputs prepared for {circuit}")
        logging.trace(f"Inputs: {inputs}")
        return inputs
