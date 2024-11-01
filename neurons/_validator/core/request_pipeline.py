from _validator.pow.proof_of_weights_handler import ProofOfWeightsHandler
import bittensor as bt
import copy
import secrets
from protocol import ProofOfWeightsSynapse, QueryZkProof

from _validator.scoring.score_manager import ScoreManager
from _validator.core.api import ValidatorAPI
from _validator.config import ValidatorConfig
from constants import BATCHED_PROOF_OF_WEIGHTS_MODEL_ID


class RequestPipeline:
    def __init__(
        self, config: ValidatorConfig, score_manager: ScoreManager, api: ValidatorAPI
    ):
        self.config = config
        self.score_manager = score_manager
        self.api = api

    def _apply_salt(
        self, request: QueryZkProof | ProofOfWeightsSynapse
    ) -> QueryZkProof | ProofOfWeightsSynapse:
        """
        Apply a salt to the request.
        """
        if (
            "validator_uid" in request["inputs"]
            and request["inputs"]["validator_uid"] is not None
        ):
            request["inputs"]["validator_uid"][-10:] = [
                secrets.randbelow(256) for _ in range(9)
            ] + [self.config.user_uid]

        if "inputs" in request and "nonce" in request["inputs"]:
            request["inputs"]["nonce"] = secrets.randbits(63)
        return request

    def prepare_requests(self, filtered_uids):
        """
        Prepare a batch of requests for the provided UIDs.

        Args:
            filtered_uids (list): List of filtered UIDs to query.

        Returns:
            list: List of prepared requests.
        """
        if self.api.external_requests_queue:
            bt.logging.info(
                f"Processing external request {self.api.external_requests_queue[-1]}"
            )
            netuid, external_request = self.api.external_requests_queue.pop()
            bt.logging.info(f"Queue size: {len(self.api.external_requests_queue)}")
            base_request = ProofOfWeightsHandler.prepare_pow_request(
                [external_request], netuid
            )
        else:
            base_request = ProofOfWeightsHandler.prepare_pow_request(
                self.score_manager.proof_of_weights_queue, self.config.subnet_uid
            )

        requests = [
            {
                "uid": uid,
                "axon": self.config.metagraph.axons[uid],
                **self._apply_salt(copy.deepcopy(base_request)),
            }
            for uid in filtered_uids
        ]

        if (
            requests
            and requests[0].get("model_id") == BATCHED_PROOF_OF_WEIGHTS_MODEL_ID
        ):
            self.score_manager.clear_proof_of_weights_queue()

        return requests
