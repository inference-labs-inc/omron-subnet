from __future__ import annotations
import torch
import bittensor as bt

from _validator.models.miner_response import MinerResponse
from _validator.utils.logging import log_scores
from _validator.utils.proof_of_weights import ProofOfWeightsItem
from _validator.utils.uid import get_queryable_uids
from constants import MAX_POW_QUEUE_SIZE, SINGLE_PROOF_OF_WEIGHTS_MODEL_ID, ONE_MINUTE
from execution_layer.verified_model_session import VerifiedModelSession
from deployment_layer.circuit_store import circuit_store
from _validator.models.request_type import RequestType
from execution_layer.circuit import CircuitEvaluationItem
from utils import with_rate_limit


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
        self.last_processed_queue_step = -1

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

    @with_rate_limit(period=ONE_MINUTE)
    def log_pow_queue_status(self):
        """Log the status of the proof of weights queue."""
        bt.logging.info(
            f"PoW Queue Status: {len(self.proof_of_weights_queue)}/{MAX_POW_QUEUE_SIZE} items "
            f"({(len(self.proof_of_weights_queue) / MAX_POW_QUEUE_SIZE) * 100:.1f}% full)"
        )

    def _update_scores_from_witness(
        self, proof_of_weights_items: list[ProofOfWeightsItem], model_id: str
    ):
        current_step = len(self.proof_of_weights_queue) >> 8
        if current_step == self.last_processed_queue_step:
            return

        bt.logging.info(
            f"Processing PoW witness generation for {len(proof_of_weights_items)} items on model {model_id}"
        )
        pow_circuit = circuit_store.get_circuit(model_id)
        if not pow_circuit:
            raise ValueError(
                f"Proof of weights circuit not found for model ID: {model_id}"
            )

        for item in proof_of_weights_items:
            if item.response_time < pow_circuit.evaluation_data.minimum_response_time:
                item.response_time = torch.max(
                    item.response_time,
                    torch.tensor(
                        pow_circuit.evaluation_data.minimum_response_time,
                        dtype=torch.float32,
                    ),
                )

            if (
                pow_circuit.evaluation_data.maximum_response_time
                <= pow_circuit.evaluation_data.minimum_response_time
            ):
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
            RequestType.RWR, ProofOfWeightsItem.to_dict_list(proof_of_weights_items)
        )
        session = VerifiedModelSession(inputs, pow_circuit)
        witness = session.generate_witness(return_content=True)
        bt.logging.info(f"Generated witness for model {model_id}")

        witness_list = witness if isinstance(witness, list) else list(witness.values())

        self._process_witness_results(witness_list, pow_circuit.settings["scaling"])

        self.last_processed_queue_step = current_step

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

        log_scores(self.scores)

    def _update_pow_queue(self, new_items: list[ProofOfWeightsItem]):
        if not new_items:
            return

        self.proof_of_weights_queue.extend(new_items)

        current_size = len(self.proof_of_weights_queue)

        if current_size > MAX_POW_QUEUE_SIZE:
            self.proof_of_weights_queue = self.proof_of_weights_queue[
                -MAX_POW_QUEUE_SIZE:
            ]

        self.log_pow_queue_status()

    def process_pow_queue(self, model_id: str) -> bool:
        """Process items in the proof of weights queue for a specific model."""
        if (
            len(self.proof_of_weights_queue) < 256
            or len(self.proof_of_weights_queue) % 256 != 0
        ):
            return False

        pow_circuit = circuit_store.get_circuit(model_id)
        if not pow_circuit:
            bt.logging.error(f"Circuit not found for model ID: {model_id}")
            return False

        items_to_process = self.proof_of_weights_queue[-256:]

        self._update_scores_from_witness(items_to_process, model_id)

        return True

    def _try_store_scores(self):
        """Attempt to store scores to disk."""
        try:
            torch.save(self.scores, self.score_path)
        except Exception as e:
            bt.logging.error(f"Error storing scores: {e}")

    def clear_proof_of_weights_queue(self):
        """Clear the proof of weights queue."""
        self.proof_of_weights_queue = []

    def update_single_score(
        self, response: MinerResponse, queryable_uids: set[int] | None = None
    ) -> None:
        """
        Update the score for a single miner based on their response.

        Args:
            response (MinerResponse): The processed response from a miner.
            queryable_uids: Optional pre-computed set of queryable UIDs.
        """
        if queryable_uids is None:
            queryable_uids = set(get_queryable_uids(self.metagraph))

        if response.uid not in queryable_uids or response.uid >= len(self.scores):
            return

        circuit = response.circuit

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

        if (
            len(self.proof_of_weights_queue) >= 256
            and len(self.proof_of_weights_queue) % 256 == 0
        ):
            self.process_pow_queue(SINGLE_PROOF_OF_WEIGHTS_MODEL_ID)

    def get_pow_queue(self) -> list[ProofOfWeightsItem]:
        """Get the current proof of weights queue."""
        return self.proof_of_weights_queue
