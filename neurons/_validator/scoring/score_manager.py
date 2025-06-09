from __future__ import annotations
import torch
import bittensor as bt

from _validator.models.miner_response import MinerResponse
from _validator.utils.logging import log_scores
from _validator.utils.proof_of_weights import ProofOfWeightsItem
from _validator.utils.uid import get_queryable_uids
from constants import (
    SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
    ONE_MINUTE,
    NUM_MINER_GROUPS,
    EPOCH_TEMPO,
    VALIDATOR_BOOST_WINDOW_BLOCKS,
    MINER_RESET_WINDOW_BLOCKS,
)
from execution_layer.verified_model_session import VerifiedModelSession
from deployment_layer.circuit_store import circuit_store
from _validator.models.request_type import RequestType
from execution_layer.circuit import CircuitEvaluationItem
from utils.rate_limiter import with_rate_limit
from utils.epoch import get_current_epoch_info, get_epoch_start_block
from _validator.competitions.competition import Competition


class ScoreManager:
    """Manages the scoring of miners."""

    def __init__(
        self,
        metagraph: bt.metagraph,
        user_uid: int,
        score_path: str,
        competition: Competition | None = None,
    ):
        """
        Initialize the ScoreManager.

        Args:
            metagraph: The metagraph of the subnet.
            user_uid: The UID of the current user.
        """
        self.metagraph = metagraph
        self.user_uid = user_uid
        self.score_path = score_path
        self.scores = self.init_scores()
        self.last_processed_queue_step = -1
        self.proof_of_weights_queue = []
        self.competition = competition
        self.last_ema_segment_per_uid = {}

    def init_scores(self) -> torch.Tensor:
        """Initialize or load existing scores."""
        bt.logging.info("Initializing validation weights")
        try:
            scores = torch.load(self.score_path, weights_only=True)
        except FileNotFoundError:
            scores = self._create_initial_scores()
        except Exception as e:
            bt.logging.error(f"Error loading scores: {e}")
            scores = self._create_initial_scores()

        bt.logging.success("Successfully initialized scores")
        log_scores(scores)
        return scores

    def _create_initial_scores(self) -> torch.Tensor:
        """Create initial scores based on metagraph data."""
        return torch.zeros(len(self.metagraph.uids), dtype=torch.float32)

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
        bt.logging.info(f"PoW Queue Status: {len(self.proof_of_weights_queue)} items")

    def _update_scores_from_witness(
        self, proof_of_weights_items: list[ProofOfWeightsItem], model_id: str
    ):
        pow_circuit = circuit_store.get_circuit(model_id)
        bt.logging.info(
            f"Processing PoW witness generation for {len(proof_of_weights_items)} items using {str(pow_circuit)}"
        )
        if not pow_circuit:
            raise ValueError(
                f"Proof of weights circuit not found for model ID: {model_id}"
            )

        inputs = pow_circuit.input_handler(
            RequestType.RWR, ProofOfWeightsItem.to_dict_list(proof_of_weights_items)
        )
        session = VerifiedModelSession(inputs, pow_circuit)
        try:
            witness = session.generate_witness(return_content=True)
            bt.logging.success(
                f"Witness for {str(pow_circuit)} generated successfully."
            )
        except Exception as e:
            bt.logging.error(f"Error generating witness: {e}")
            return

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

        log_scores(self.scores)
        self._try_store_scores()

    def _update_pow_queue(self, new_items: list[ProofOfWeightsItem]):
        if not new_items:
            return

        self.proof_of_weights_queue.extend(new_items)
        self.log_pow_queue_status()

    def process_pow_queue(self, model_id: str) -> bool:
        """Process items in the proof of weights queue for a specific model."""
        if (
            len(self.proof_of_weights_queue) < 256
            or len(self.proof_of_weights_queue) % 256 != 0
        ):
            return False

        current_step = len(self.proof_of_weights_queue) >> 8
        if current_step == self.last_processed_queue_step:
            return False

        pow_circuit = circuit_store.get_circuit(model_id)
        if not pow_circuit:
            bt.logging.error(f"Circuit not found for model ID: {model_id}")
            return False

        items_to_process = self.proof_of_weights_queue[-256:]
        self._update_scores_from_witness(items_to_process, model_id)
        self.last_processed_queue_step = current_step

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

    @with_rate_limit(period=ONE_MINUTE * 5)
    def _get_last_bonds_submissions(self) -> int:
        """Get the latest bonds submissions."""
        return self.metagraph.subtensor.substrate.query_map(
            "Commitments",
            "LastBondsReset",
            params=[self.metagraph.netuid],
        )

    def _miner_missed_reset(
        self,
        uid: int,
        miner_group: int,
        current_epoch: int,
        blocks_until_next_epoch: int,
    ) -> bool:
        """
        Check if a miner missed their required reset submission during their tempo.
        Returns True if the miner was supposed to reset but didn't.
        """
        try:
            last_bonds_submissions = self._get_last_bonds_submissions()

            if self.metagraph.hotkeys[uid] not in last_bonds_submissions:
                return True

            last_reset_block = last_bonds_submissions[self.metagraph.hotkeys[uid]]

            if (
                current_epoch % NUM_MINER_GROUPS == miner_group
                and blocks_until_next_epoch <= MINER_RESET_WINDOW_BLOCKS
            ):
                most_recent_group_epoch = current_epoch
            else:
                epochs_since_last = (
                    current_epoch % NUM_MINER_GROUPS - miner_group
                ) % NUM_MINER_GROUPS
                most_recent_group_epoch = current_epoch - epochs_since_last

            if most_recent_group_epoch >= 0:
                epoch_start_block = get_epoch_start_block(
                    most_recent_group_epoch, self.metagraph.netuid
                )
                epoch_end_block = epoch_start_block + EPOCH_TEMPO
                if (
                    last_reset_block < (epoch_end_block - MINER_RESET_WINDOW_BLOCKS)
                    or last_reset_block > epoch_end_block
                ):
                    return True

            return False
        except Exception as e:
            bt.logging.error(f"Error checking reset status for miner {uid}: {e}")
            return False

    @with_rate_limit(period=ONE_MINUTE * 5)
    def process_non_queryable_scores(self, queryable_uids: set[int], max_score: float):
        """
        Decay scores for non-queryable UIDs.
        """
        for uid in range(len(self.scores)):
            if uid not in queryable_uids:
                hotkey = self.metagraph.hotkeys[uid]
                if not (self.competition and hotkey in self.competition.miner_states):
                    self.scores[uid] = 0
                elif self.competition and hotkey in self.competition.miner_states:
                    pow_item = ProofOfWeightsItem.for_competition(
                        uid=uid,
                        maximum_score=max_score,
                        competition_score=self.competition.miner_states[
                            hotkey
                        ].sota_relative_score,
                        block_number=self.metagraph.block.item(),
                        validator_uid=self.user_uid,
                    )
                    self._update_pow_queue([pow_item])

    def update_single_score(
        self,
        response: MinerResponse,
        queryable_uids: set[int] | None = None,
    ) -> None:
        """
        Update the score for a single miner based on their response.

        Args:
            response (MinerResponse): The processed response from a miner.
            queryable_uids: Optional pre-computed set of queryable UIDs.
        """
        if queryable_uids is None:
            queryable_uids = set(get_queryable_uids(self.metagraph))

        circuit = response.circuit

        current_block = self.metagraph.subtensor.get_current_block()
        miner_group = response.uid % NUM_MINER_GROUPS
        current_epoch, blocks_until_next_epoch, _ = get_current_epoch_info(
            current_block, self.metagraph.netuid
        )

        if self._miner_missed_reset(
            response.uid, miner_group, current_epoch, blocks_until_next_epoch
        ):
            bt.logging.warning(
                f"Miner {response.uid} missed required reset submission, marking as unverified"
            )
            response.verification_result = False

        if self.scores[response.uid] is not None:
            last_ema_epoch = self.last_ema_segment_per_uid.get(response.uid, -1)

            if last_ema_epoch != current_epoch:
                active_group = current_epoch % NUM_MINER_GROUPS
                # -1 for commit reveal delay; max bonds impact at t0
                if (
                    miner_group == active_group - 1
                    and blocks_until_next_epoch <= VALIDATOR_BOOST_WINDOW_BLOCKS
                ):
                    self.scores[response.uid] = self.scores[response.uid] * 1.2
                else:
                    self.scores[response.uid] = self.scores[response.uid] * 0.99

                self.last_ema_segment_per_uid[response.uid] = current_epoch

        evaluation_data = CircuitEvaluationItem(
            circuit=circuit,
            uid=response.uid,
            minimum_response_time=circuit.evaluation_data.minimum_response_time,
            proof_size=response.proof_size,
            response_time=response.response_time,
            score=self.scores[response.uid],
            verification_result=response.verification_result,
        )
        circuit.evaluation_data.update(evaluation_data)

        max_score = 1 / len(self.scores)
        self.process_non_queryable_scores(queryable_uids, max_score)

        pow_item = ProofOfWeightsItem.from_miner_response(
            response,
            max_score,
            self.scores[response.uid],
            circuit.evaluation_data.maximum_response_time,
            circuit.evaluation_data.minimum_response_time,
            self.metagraph.block.item(),
            self.user_uid,
            0,
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

    def remove_processed_items(self, count: int):
        if count <= 0:
            return
        self.proof_of_weights_queue = self.proof_of_weights_queue[count:]
