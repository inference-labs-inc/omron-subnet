from __future__ import annotations
import traceback
import random
from _validator.models.completed_proof_of_weights import CompletedProofOfWeightsItem

from _validator.models.miner_response import MinerResponse
from _validator.scoring.score_manager import ScoreManager
from _validator.utils.logging import log_responses, log_system_metrics
from deployment_layer.circuit_store import circuit_store


# trunk-ignore (pylint/E0611)
from bittensor import logging
from _validator.utils.proof_of_weights import save_proof_of_weights

from constants import BATCHED_PROOF_OF_WEIGHTS_MODEL_ID
from execution_layer.verified_model_session import VerifiedModelSession
from utils import wandb_logger


class ResponseProcessor:
    def __init__(self, metagraph, score_manager: ScoreManager, user_uid):
        self.metagraph = metagraph
        self.score_manager = score_manager
        self.user_uid = user_uid
        self.proof_batches_queue = []
        self.completed_proof_of_weights_queue: list[CompletedProofOfWeightsItem] = []

    def process_responses(self, responses: list[dict]) -> list[MinerResponse]:
        processed_responses = [self.process_single_response(r) for r in responses]
        log_responses(processed_responses)
        response_times = [
            r.response_time
            for r in processed_responses
            if r.response_time is not None and r.verification_result
        ]
        verified_count = sum(1 for r in processed_responses if r.verification_result)
        log_system_metrics(response_times, verified_count)

        if not processed_responses[0].model_id == BATCHED_PROOF_OF_WEIGHTS_MODEL_ID:
            return processed_responses

        verified_batched_responses = [
            r
            for r in processed_responses
            if r.verification_result and r.proof_json is not None
        ]
        if verified_batched_responses:
            # trunk-ignore(bandit/B311)
            selected_response = random.choice(verified_batched_responses)
            logging.debug(
                f"Selected Proof of Weights from UID: {selected_response.uid} to use "
                "as batched proof of weights for this interval."
            )
            save_proof_of_weights(selected_response.public_json, selected_response.proof_json)  # type: ignore
            self.completed_proof_of_weights_queue.append(
                CompletedProofOfWeightsItem(
                    selected_response.public_json,
                    selected_response.proof_json,
                    BATCHED_PROOF_OF_WEIGHTS_MODEL_ID,
                )
            )
        else:
            logging.error("No valid batched proof of weights found.")
            wandb_logger.safe_log(
                {
                    "valid_batched_proof_of_weights": False,
                }
            )

        return processed_responses

    def process_single_response(self, response: dict) -> MinerResponse:
        miner_response = MinerResponse.from_raw_response(response)
        if miner_response.proof_json is None:
            logging.debug(
                f"Miner at UID: {miner_response.uid} failed to provide a valid proof. "
                f"Response from miner: {miner_response.raw}"
            )
        elif miner_response.proof_json:
            logging.debug(f"Attempting to verify proof for UID: {miner_response.uid}")
            try:
                verification_result = self.verify_proof_string(
                    miner_response, response["inputs"]
                )
                miner_response.set_verification_result(verification_result)
                if not verification_result:
                    logging.warning(
                        f"Miner at UID: {miner_response.uid} provided a proof, but verification failed."
                    )
            except Exception as e:
                logging.warning(
                    f"Unable to verify proof for UID: {miner_response.uid}. Error: {e}"
                )
                traceback.print_exc()

            if miner_response.verification_result:
                logging.success(
                    f"Miner at UID: {miner_response.uid} provided a valid proof "
                    f"in {miner_response.response_time} seconds."
                )
        return miner_response

    def verify_proof_string(
        self, response: MinerResponse, validator_inputs: list[float] | dict
    ) -> bool:
        if not response.proof_json or not response.public_json:
            logging.error(f"Proof or public json not found for UID: {response.uid}")
            return False
        try:
            inference_session = VerifiedModelSession(
                validator_inputs, circuit_store.get_circuit(str(response.model_id))
            )
            res: bool = inference_session.verify_proof(
                response.public_json, response.proof_json
            )
            return res
        except Exception as e:
            raise e
