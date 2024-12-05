import torch
import bittensor as bt
from _validator.models.miner_response import MinerResponse
from _validator.utils.logging import log_scores
from _validator.utils.proof_of_weights import ProofOfWeightsItem
from constants import (
    MAXIMUM_SCORE_MEDIAN_SAMPLE,
    MINIMUM_SCORE_SHIFT,
    SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
    VALIDATOR_REQUEST_TIMEOUT_SECONDS,
)
from execution_layer.verified_model_session import VerifiedModelSession
from deployment_layer.circuit_store import circuit_store
from _validator.models.request_type import RequestType


class ScoreManager:
    """Manages the scoring of miners."""

    def __init__(self, metagraph, user_uid):
        """
        Initialize the ScoreManager.

        Args:
            metagraph: The metagraph of the subnet.
            user_uid: The UID of the current user.
        """
        self.metagraph = metagraph
        self.user_uid = user_uid
        self.scores = self.init_scores()
        self.proof_of_weights_queue = []

    def init_scores(self):
        """Initialize or load existing scores."""
        bt.logging.info("Initializing validation weights")
        try:
            scores = torch.load("scores.pt")
        except FileNotFoundError:
            scores = self._create_initial_scores()
        except Exception as e:
            bt.logging.error(f"Error loading scores: {e}")
            scores = torch.zeros_like(self.metagraph.S, dtype=torch.float32)

        bt.logging.success("Successfully set up scores")
        log_scores(scores)
        return scores

    def _create_initial_scores(self):
        """Create initial scores based on metagraph data."""
        # Depending on how bittensor was installed, metagraph may be tensor or ndarray
        total_stake = (
            self.metagraph.S
            if isinstance(self.metagraph.S, torch.Tensor)
            else torch.tensor(self.metagraph.S)
        )
        scores = torch.zeros_like(total_stake, dtype=torch.float32)
        return scores * torch.Tensor(
            [
                # trunk-ignore(bandit/B104)
                self.metagraph.neurons[uid].axon_info.ip != "0.0.0.0"
                for uid in self.metagraph.uids
            ]
        )

    def update_scores(self, responses: list[MinerResponse]) -> None:
        """
        Update scores based on miner responses.

        Args:
            responses: List of MinerResponse objects.
        """
        if not responses or self.scores is None:
            bt.logging.error("No responses or scores not initialized. Skipping update.")
            return

        max_score = 1 / len(self.scores)
        responses = self._add_missing_responses(responses)

        sorted_filtered_times = self._get_sorted_filtered_times(responses)
        median_max_response_time, min_response_time = (
            self._calculate_response_time_metrics(sorted_filtered_times)
        )

        proof_of_weights_items = self._create_pow_items(
            responses, max_score, median_max_response_time, min_response_time
        )

        self._update_scores_from_witness(proof_of_weights_items)
        self._update_pow_queue(proof_of_weights_items)

        log_scores(self.scores)
        self._try_store_scores()

    def _add_missing_responses(
        self, responses: list[MinerResponse]
    ) -> list[MinerResponse]:
        """Add missing responses for all UIDs not present in the original responses."""
        all_uids = set(range(len(self.scores) - 1))
        response_uids = set(r.uid for r in responses)
        missing_uids = all_uids - response_uids
        responses.extend(MinerResponse.empty(uid) for uid in missing_uids)
        return responses

    def _get_sorted_filtered_times(self, responses: list[MinerResponse]) -> list[float]:
        """Get sorted list of valid response times."""
        return sorted(
            r.response_time
            for r in responses
            if r.verification_result and r.response_time > 0
        )

    def _calculate_response_time_metrics(
        self, sorted_filtered_times: list[float]
    ) -> tuple[float, float]:
        """Calculate median max and minimum response times."""
        if not sorted_filtered_times:
            return VALIDATOR_REQUEST_TIMEOUT_SECONDS, 0

        sample_size = max(
            int(len(sorted_filtered_times) * MAXIMUM_SCORE_MEDIAN_SAMPLE), 1
        )
        median_max_response_time = torch.clamp(
            torch.median(torch.tensor(sorted_filtered_times[-sample_size:])),
            0,
            VALIDATOR_REQUEST_TIMEOUT_SECONDS,
        ).item()

        min_response_time = (
            torch.clamp(
                torch.min(torch.tensor(sorted_filtered_times)),
                0,
                VALIDATOR_REQUEST_TIMEOUT_SECONDS,
            ).item()
            - MINIMUM_SCORE_SHIFT
        )

        return median_max_response_time, min_response_time

    def _create_pow_items(
        self,
        responses: list[MinerResponse],
        max_score: float,
        median_max_response_time: float,
        min_response_time: float,
    ) -> list[ProofOfWeightsItem]:
        """Create ProofOfWeightsItems from responses."""
        return [
            ProofOfWeightsItem.from_miner_response(
                response,
                max_score,
                self.scores[response.uid],
                median_max_response_time,
                min_response_time,
                self.metagraph.block.item(),
                self.user_uid,
            )
            for response in responses
        ]

    def _update_scores_from_witness(
        self, proof_of_weights_items: list[ProofOfWeightsItem]
    ):
        """Update scores based on the witness generated from proof of weights items."""
        pow_circuit = circuit_store.get_circuit(SINGLE_PROOF_OF_WEIGHTS_MODEL_ID)
        if not pow_circuit:
            raise ValueError(
                f"Proof of weights circuit not found for model ID: {SINGLE_PROOF_OF_WEIGHTS_MODEL_ID}"
            )

        padded_items = ProofOfWeightsItem.pad_items(proof_of_weights_items, 256)
        for item in padded_items:
            if item.response_time < item.minimum_response_time:
                bt.logging.warning(
                    f"Response time {item.response_time.item()} is less than minimum"
                    f" {item.minimum_response_time.item()} for UID {item.miner_uid.item()}"
                )

                item.response_time = torch.max(
                    item.response_time, item.minimum_response_time
                )

            # Ensure there is > 0 spread between min and max response times (usually during testing)
            if item.maximum_response_time <= item.minimum_response_time:
                bt.logging.warning(
                    f"No spread between min and max response times for UID {item.miner_uid.item()}"
                )
                item.maximum_response_time = item.minimum_response_time + 1

        inputs = pow_circuit.input_handler(
            RequestType.RWR, ProofOfWeightsItem.to_dict_list(padded_items)
        )

        session = VerifiedModelSession(inputs, pow_circuit)
        witness = session.generate_witness(return_content=True)
        witness_list = witness if isinstance(witness, list) else list(witness.values())

        self._process_witness_results(witness_list, pow_circuit.settings["scaling"])

        session.end()

    def _process_witness_results(self, witness: list, scaling: int):
        """Process the results from the witness."""
        scores = torch.div(
            torch.tensor([float(w) for w in witness[1:257]]), scaling
        ).tolist()
        miner_uids = [int(float(w)) for w in witness[513:769]]

        bt.logging.debug(
            f"Proof of weights scores: {scores} for miner UIDs: {miner_uids}, existing scores: {self.scores}"
        )

        for uid, score in zip(miner_uids, scores):
            if uid < 0 or uid >= len(self.scores):
                continue
            self.scores[uid] = float(score)
            bt.logging.debug(f"Updated score for UID {uid}: {score}")

    def _update_pow_queue(self, new_items: list[ProofOfWeightsItem]):
        """Update the proof of weights queue with new items."""
        self.proof_of_weights_queue = ProofOfWeightsItem.merge_items(
            self.proof_of_weights_queue, new_items
        )

    def _try_store_scores(self):
        """Attempt to store scores to disk."""
        try:
            torch.save(self.scores, "scores.pt")
        except Exception as e:
            bt.logging.info(f"Error storing scores: {e}")

    def get_proof_of_weights_queue(self):
        """Return the current proof of weights queue."""
        return self.proof_of_weights_queue

    def clear_proof_of_weights_queue(self):
        """Clear the proof of weights queue."""
        self.proof_of_weights_queue = []

    def sync_scores_uids(self, uids: list[int]):
        """
        If there are more uids than scores, add more weights.
        """
        if len(uids) > len(self.scores):
            bt.logging.trace(
                f"Scores length: {len(self.scores)}, UIDs length: {len(uids)}. Adding more weights"
            )
            size_difference = len(uids) - len(self.scores)
            new_scores = torch.zeros(size_difference, dtype=torch.float32)
            self.scores = torch.cat((self.scores, new_scores))
