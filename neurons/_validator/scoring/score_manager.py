from __future__ import annotations
import torch
import bittensor as bt

from _validator.models.miner_response import MinerResponse
from _validator.utils.logging import log_scores
from _validator.utils.proof_of_weights import ProofOfWeightsItem
from _validator.utils.uid import get_queryable_uids
from constants import (
    MAX_POW_QUEUE_SIZE,
)
from execution_layer.verified_model_session import VerifiedModelSession
from deployment_layer.circuit_store import circuit_store
from _validator.models.request_type import RequestType
from execution_layer.circuit import CircuitEvaluationItem


class ScoreManager:
    """Manages the scoring of miners."""

    def __init__(self, metagraph: bt.metagraph, user_uid: int, score_path: str):
        """
        Initialize the ScoreManager.

        Args:
            metagraph: The metagraph of the subnet.
            user_uid: The UID of the current user.
        """
        self.metagraph = metagraph
        self.user_uid = user_uid
        self.score_path = score_path
        self.scores = torch.Tensor([])

        self.proof_of_weights_queue = []

    def init_scores(self, model_id: str) -> torch.Tensor:
        """Initialize or load existing scores."""
        bt.logging.info("Initializing validation weights")
        try:
            scores = torch.load(self.score_path, weights_only=True)
        except FileNotFoundError:
            scores = self._create_initial_scores()
        except Exception as e:
            bt.logging.error(f"Error loading scores: {e}")
            scores = self._create_initial_scores()

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
        queryable_uids = set(get_queryable_uids(self.metagraph))
        return scores * torch.Tensor(
            [uid in queryable_uids for uid in self.metagraph.uids]
        )

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
            queryable_uids = set(get_queryable_uids(self.metagraph))
            new_scores = new_scores * torch.Tensor(
                [
                    uid in queryable_uids
                    for uid in self.metagraph.uids[len(self.scores) :]
                ]
            )
            self.scores = torch.cat((self.scores, new_scores))

    def _update_scores_single_model(
        self, responses: list[MinerResponse], model_id: str
    ):
        """
        Update scores for a single model.
        """
        max_score = 1 / len(self.scores)
        circuit = circuit_store.get_circuit(model_id)

        # Update evaluation data with new responses
        for response in responses:
            if response.verification_result and response.response_time > 0:
                circuit.evaluation_data.update(
                    CircuitEvaluationItem(
                        circuit_id=model_id,
                        uid=response.uid,
                        minimum_response_time=circuit.evaluation_data.minimum_response_time,
                        maximum_response_time=circuit.evaluation_data.maximum_response_time,
                        proof_size=response.proof_size,
                        response_time=response.response_time,
                        score=self.scores[response.uid],
                        verification_result=response.verification_result,
                    )
                )

        # Get metrics from updated evaluation data
        median_max_response_time = circuit.evaluation_data.maximum_response_time
        min_response_time = circuit.evaluation_data.minimum_response_time

        proof_of_weights_items = self._create_pow_items(
            responses, max_score, median_max_response_time, min_response_time, model_id
        )

        self._update_scores_from_witness(proof_of_weights_items, model_id)
        self._update_pow_queue(proof_of_weights_items)

        log_scores(self.scores)
        self._try_store_scores()

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
                self.scores[response.uid],
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

        # Always use 256 batch size for witness generation
        batch_size = 256
        padded_items = ProofOfWeightsItem.pad_items(proof_of_weights_items, batch_size)

        # Use circuit evaluation data for response times
        for item in padded_items:
            if item.response_time < pow_circuit.evaluation_data.minimum_response_time:
                bt.logging.warning(
                    f"Response time {item.response_time.item()} is less than minimum"
                    f" {pow_circuit.evaluation_data.minimum_response_time} for UID {item.miner_uid.item()}"
                )

                item.response_time = torch.max(
                    item.response_time,
                    torch.tensor(
                        pow_circuit.evaluation_data.minimum_response_time,
                        dtype=torch.float32,
                    ),
                )

            # Ensure there is > 0 spread between min and max response times
            if (
                pow_circuit.evaluation_data.maximum_response_time
                <= pow_circuit.evaluation_data.minimum_response_time
            ):
                bt.logging.warning(
                    f"No spread between min and max response times for UID {item.miner_uid.item()}"
                )
                item.maximum_response_time = torch.tensor(
                    pow_circuit.evaluation_data.minimum_response_time + 1,
                    dtype=torch.float32,
                )
            else:
                item.maximum_response_time = torch.tensor(
                    pow_circuit.evaluation_data.maximum_response_time,
                    dtype=torch.float32,
                )

        inputs = pow_circuit.input_handler(
            RequestType.RWR, ProofOfWeightsItem.to_dict_list(padded_items)
        )

        session = VerifiedModelSession(inputs, pow_circuit)
        witness = session.generate_witness(return_content=True)
        witness_list = witness if isinstance(witness, list) else list(witness.values())

        # Process witness results based on batch size
        scores_end = batch_size + 1
        uids_start = batch_size * 2 + 1
        uids_end = uids_start + batch_size

        self._process_witness_results(
            witness_list[1:scores_end],
            witness_list[uids_start:uids_end],
            pow_circuit.settings["scaling"],
            model_id,
        )

        session.end()

    def _process_witness_results(
        self, scores: list, miner_uids: list, scaling: int, model_id: str
    ):
        """Process the results from the witness."""
        scores = torch.div(torch.tensor([float(w) for w in scores]), scaling).tolist()
        miner_uids = [int(float(w)) for w in miner_uids]

        bt.logging.debug(
            f"Proof of weights scores: {scores} for miner UIDs: {miner_uids} for "
            f"model ID: {model_id}, existing scores: {self.scores}"
        )

        for uid, score in zip(miner_uids, scores):
            if uid >= len(self.scores):
                continue
            self.scores[uid] = float(score)
            bt.logging.debug(f"Updated score for UID {uid}: {score}")

    def _update_pow_queue(self, new_items: list[ProofOfWeightsItem]):
        """Update the proof of weights queue with items, maintaining size limit."""
        merged = ProofOfWeightsItem.merge_items(self.proof_of_weights_queue, new_items)
        self.proof_of_weights_queue = merged[-MAX_POW_QUEUE_SIZE:]

    def _try_store_scores(self):
        """Attempt to store scores to disk."""
        try:
            torch.save(self.scores, self.score_path)
        except Exception as e:
            bt.logging.error(f"Error storing scores: {e}")

    def clear_proof_of_weights_queue(self):
        """Clear the proof of weights queue."""
        self.proof_of_weights_queue = []

    def update_scores(
        self, responses: list[MinerResponse], queryable_uids: set[int] | None = None
    ) -> None:
        """
        Update scores based on miner responses.

        Args:
            responses: List of MinerResponse objects.
            queryable_uids: Optional pre-computed set of queryable UIDs.
        """
        if not responses:
            bt.logging.error("No responses. Skipping update.")
            return

        # Use provided queryable_uids or compute if not provided
        if queryable_uids is None:
            queryable_uids = set(get_queryable_uids(self.metagraph))

        # Pre-filter responses for valid UIDs
        valid_responses = [
            r for r in responses if r.uid in queryable_uids and r.uid < len(self.scores)
        ]

        if not valid_responses:
            bt.logging.warning(
                "No valid responses after UID filtering. Skipping update."
            )
            return

        for model_id in circuit_store.list_circuits():
            circuit = circuit_store.get_circuit(model_id)
            if not circuit:
                bt.logging.error(f"Circuit not found for model ID: {model_id}")
                continue

            responses_for_model = [
                r for r in valid_responses if r.circuit.id == model_id
            ]
            if responses_for_model:
                self._update_scores_single_model(responses_for_model, model_id)

    def update_single_score(
        self, response: MinerResponse, queryable_uids: set[int] | None = None
    ) -> None:
        """
        Update the score for a single miner based on their response.

        Args:
            response (MinerResponse): The processed response from a miner.
            queryable_uids: Optional pre-computed set of queryable UIDs.
        """
        # Use provided queryable_uids or compute if not provided
        if queryable_uids is None:
            queryable_uids = set(get_queryable_uids(self.metagraph))

        # Skip if UID isn't queryable
        if response.uid not in queryable_uids or response.uid >= len(self.scores):
            return

        circuit = response.circuit

        # First update circuit evaluation data
        evaluation_data = CircuitEvaluationItem(
            circuit_id=circuit.id,
            uid=response.uid,
            minimum_response_time=circuit.evaluation_data.minimum_response_time,
            maximum_response_time=circuit.evaluation_data.maximum_response_time,
            proof_size=response.proof_size,
            response_time=response.response_time,
            score=self.scores[response.uid],
            verification_result=response.verification_result,
        )
        circuit.evaluation_data.update(evaluation_data)

        # Then create and add PoW item
        max_score = 1 / len(self.scores)
        pow_item = ProofOfWeightsItem.from_miner_response(
            response,
            max_score,
            self.scores[response.uid],
            circuit.evaluation_data.maximum_response_time,
            circuit.evaluation_data.minimum_response_time,
            self.metagraph.block.item(),
            self.user_uid,
        )
        self._update_pow_queue([pow_item])
