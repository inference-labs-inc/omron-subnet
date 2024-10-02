import torch
import bittensor as bt
from _validator.models.miner_response import MinerResponse
from _validator.scoring.reward import (
    FLATTENING_COEFFICIENT,
    MAXIMUM_RESPONSE_TIME_DECIMAL,
    PROOF_SIZE_THRESHOLD,
    PROOF_SIZE_WEIGHT,
    RATE_OF_DECAY,
    RATE_OF_RECOVERY,
    RESPONSE_TIME_WEIGHT,
)
from _validator.utils.logging import log_scores
from _validator.utils.proof_of_weights import ProofOfWeightsItem
from constants import (
    MAXIMUM_SCORE_MEDIAN_SAMPLE,
    MINIMUM_SCORE_SHIFT,
    VALIDATOR_REQUEST_TIMEOUT_SECONDS,
)
from execution_layer.verified_model_session import VerifiedModelSession
from deployment_layer.circuit_store import circuit_store


class ScoreManager:
    """Manages the scoring of miners."""

    def __init__(self, metagraph: bt.metagraph, user_uid: int):
        """
        Initialize the ScoreManager.

        Args:
            metagraph: The metagraph of the subnet.
            user_uid: The UID of the current user.
        """
        self.metagraph: bt.metagraph = metagraph
        self.user_uid: int = user_uid
        self.score_dict: dict[torch.Tensor] = {
            model_id: self.init_scores(model_id) for model_id in circuit_store.list_circuits()
        }
        self.proof_of_weights_queue = []

    def init_scores(self, model_id: str) -> torch.Tensor:
        """Initialize or load existing scores."""
        bt.logging.info("Initializing validation weights")
        try:
            scores = torch.load(f"scores_{model_id}.pt")
        except FileNotFoundError:
            scores = self._create_initial_scores()
        except Exception as e:
            bt.logging.error(f"Error loading scores: {e}")
            scores = torch.zeros_like(self.metagraph.S, dtype=torch.float32)

        bt.logging.success(f"Successfully set up scores for model ID: {model_id}")
        log_scores(scores)
        return scores

    def _create_initial_scores(self) -> torch.Tensor:
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
        if not responses:
            bt.logging.error("No responses. Skipping update.")
            return

        for model_id in circuit_store.list_circuits():
            responses_for_model = [r for r in responses if r.model_id == model_id]
            if responses_for_model:
                self._update_scores_single_model(responses_for_model, model_id)

    def _update_scores_single_model(self, responses: list[MinerResponse], model_id: str):
        """
        Update scores for a single model.
        """
        max_score = 1 / len(self.score_dict[model_id])
        responses = self._add_missing_responses(responses, model_id)

        sorted_filtered_times = self._get_sorted_filtered_times(responses)
        median_max_response_time, min_response_time = (
            self._calculate_response_time_metrics(sorted_filtered_times)
        )

        proof_of_weights_items = self._create_pow_items(
            responses, max_score, median_max_response_time, min_response_time, model_id
        )

        self._update_scores_from_witness(proof_of_weights_items, model_id)
        self._update_pow_queue(proof_of_weights_items)

        log_scores(self.score_dict[model_id])
        self._try_store_scores(model_id)

    def _add_missing_responses(
        self, responses: list[MinerResponse], model_id: str
    ) -> list[MinerResponse]:
        """Add missing responses for all UIDs not present in the original responses."""
        all_uids = set(range(len(self.score_dict[model_id]) - 1))
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
        model_id: str,
    ) -> list[ProofOfWeightsItem]:
        """Create ProofOfWeightsItems from responses."""
        return [
            ProofOfWeightsItem.from_miner_response(
                response,
                max_score,
                self.score_dict[model_id][response.uid],
                median_max_response_time,
                min_response_time,
                self.metagraph.block.item(),
                self.user_uid,
            )
            for response in responses
        ]

    def _update_scores_from_witness(
        self, proof_of_weights_items: list[ProofOfWeightsItem], model_id: str
    ):
        """Update scores based on the witness generated from proof of weights items."""
        pow_circuit = circuit_store.get_circuit(model_id)
        if not pow_circuit:
            raise ValueError(
                f"Proof of weights circuit not found for model ID: {model_id}"
            )

        padded_items = ProofOfWeightsItem.pad_items(proof_of_weights_items, 256)
        for item in padded_items:
            if item.response_time < item.min_response_time:
                bt.logging.warning(
                    f"Response time {item.response_time.item()} is less than minimum"
                    f" {item.min_response_time.item()} for UID {item.uid.item()}"
                )

                item.response_time = torch.max(
                    item.response_time, item.min_response_time
                )

            # Ensure there is > 0 spread between min and max response times (usually during testing)
            if item.median_max_response_time <= item.min_response_time:
                bt.logging.warning(
                    f"No spread between min and max response times for UID {item.uid.item()}"
                )
                item.median_max_response_time = item.min_response_time + 1
        inputs = self._prepare_pow_inputs(padded_items, pow_circuit.settings["scaling"])

        session = VerifiedModelSession(inputs, pow_circuit)
        witness = session.generate_witness(return_content=True)
        witness_list = witness if isinstance(witness, list) else list(witness.values())

        self._process_witness_results(witness_list, pow_circuit.settings["scaling"], model_id)

        session.end()

    def _prepare_pow_inputs(
        self, items: list[ProofOfWeightsItem], scaling: int
    ) -> dict:
        """Prepare inputs for the proof of weights circuit."""
        return {
            "RATE_OF_DECAY": int(RATE_OF_DECAY * scaling),
            "RATE_OF_RECOVERY": int(RATE_OF_RECOVERY * scaling),
            "FLATTENING_COEFFICIENT": int(FLATTENING_COEFFICIENT * scaling),
            "PROOF_SIZE_WEIGHT": int(PROOF_SIZE_WEIGHT * scaling),
            "PROOF_SIZE_THRESHOLD": int(PROOF_SIZE_THRESHOLD * scaling),
            "RESPONSE_TIME_WEIGHT": int(RESPONSE_TIME_WEIGHT * scaling),
            "MAXIMUM_RESPONSE_TIME_DECIMAL": int(
                MAXIMUM_RESPONSE_TIME_DECIMAL * scaling
            ),
            "maximum_score": [int(item.max_score.item() * scaling) for item in items],
            "previous_score": [
                (
                    int(item.previous_score.item() * scaling)
                    if not torch.isnan(item.previous_score)
                    else 0
                )
                for item in items
            ],
            "verified": [item.verification_result.item() for item in items],
            "proof_size": [int(item.proof_size.item() * scaling) for item in items],
            "response_time": [
                int(item.response_time.item() * scaling) for item in items
            ],
            "maximum_response_time": [
                int(item.median_max_response_time.item() * scaling) for item in items
            ],
            "minimum_response_time": [
                int(item.min_response_time.item() * scaling) for item in items
            ],
            "block_number": [item.block_number.item() for item in items],
            "validator_uid": [item.validator_uid.item() for item in items],
            "miner_uid": [item.uid.item() for item in items],
            "scaling": scaling,
        }

    def _process_witness_results(self, witness: list, scaling: int, model_id: str):
        """Process the results from the witness."""
        scores = torch.div(
            torch.tensor([float(w) for w in witness[1:257]]), scaling
        ).tolist()
        miner_uids = [int(float(w)) for w in witness[513:769]]

        bt.logging.debug(
            f"Proof of weights scores: {scores} for miner UIDs: {miner_uids} for model ID: {model_id}, existing scores: {self.score_dict[model_id]}"
        )

        for uid, score in zip(miner_uids, scores):
            if uid >= len(self.score_dict[model_id]):
                continue
            self.score_dict[model_id][uid] = float(score)
            bt.logging.debug(f"Updated score for UID {uid}: {score}")

    def _update_pow_queue(self, new_items: list[ProofOfWeightsItem]):
        """Update the proof of weights queue with new items."""
        self.proof_of_weights_queue = ProofOfWeightsItem.merge_items(
            self.proof_of_weights_queue, new_items
        )

    def _try_store_scores(self, model_id: str):
        """Attempt to store scores to disk."""
        try:
            torch.save(self.score_dict[model_id], f"scores_{model_id}.pt")
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
        for model_id in circuit_store.list_circuits():
            if len(uids) > len(self.score_dict[model_id]):
                bt.logging.trace(
                    f"Scores length: {len(self.score_dict[model_id])}, UIDs length: {len(uids)}. Adding more weights"
                )
                size_difference = len(uids) - len(self.score_dict[model_id])
                new_scores = torch.zeros(size_difference, dtype=torch.float32)
                self.score_dict[model_id] = torch.cat((self.score_dict[model_id], new_scores))
