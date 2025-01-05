from __future__ import annotations
from _validator.pow.proof_of_weights_handler import ProofOfWeightsHandler
import bittensor as bt
import random
from protocol import ProofOfWeightsSynapse, QueryZkProof

from _validator.scoring.score_manager import ScoreManager
from _validator.api import ValidatorAPI
from _validator.config import ValidatorConfig
from constants import (
    BATCHED_PROOF_OF_WEIGHTS_MODEL_ID,
    CIRCUIT_WEIGHTS,
    SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
)
from deployment_layer.circuit_store import circuit_store
from execution_layer.circuit import Circuit, CircuitType
from execution_layer.generic_input import GenericInput
from _validator.utils.hash_guard import HashGuard
from _validator.core.request import Request
from utils.wandb_logger import safe_log
from _validator.models.request_type import RequestType


class RequestPipeline:
    def __init__(
        self, config: ValidatorConfig, score_manager: ScoreManager, api: ValidatorAPI
    ):
        self.config = config
        self.score_manager = score_manager
        self.api = api
        self.hash_guard = HashGuard()

    def prepare_requests(self, filtered_uids) -> list[Request]:
        """
        Prepare requests for the current validation step.
        This includes both regular benchmark requests and any external requests.

        Args:
            filtered_uids (list): List of UIDs to send requests to.

        Returns:
            list[Request]: List of prepared requests.
        """
        if len(filtered_uids) == 0:
            bt.logging.error("No UIDs to query")
            return []

        if self.api.external_requests_queue:
            return self._prepare_real_world_requests(filtered_uids)
        return self._prepare_benchmark_requests(filtered_uids)

    def _prepare_real_world_requests(self, filtered_uids: list[int]) -> list[Request]:
        external_request = self.api.external_requests_queue.pop()
        requests = []

        for uid in filtered_uids:
            synapse = self.get_synapse_request(
                RequestType.RWR, external_request.circuit, external_request
            )

            if isinstance(synapse, ProofOfWeightsSynapse):
                input_data = synapse.inputs
            else:
                input_data = synapse.query_input["public_inputs"]

            try:
                self.hash_guard.check_hash(input_data)
            except Exception as e:
                bt.logging.error(f"Hash already exists: {e}")
                safe_log({"hash_guard_error": 1})
                continue

            request = Request(
                uid=uid,
                axon=self.config.metagraph.axons[uid],
                synapse=synapse,
                circuit=external_request.circuit,
                request_type=RequestType.RWR,
                inputs=GenericInput(RequestType.RWR, input_data),
                request_hash=external_request.hash,
            )
            requests.append(request)
        return requests

    def _prepare_benchmark_requests(self, filtered_uids: list[int]) -> list[Request]:
        circuit = self.select_circuit_for_benchmark()
        if circuit is None:
            bt.logging.error("No circuit selected")
            return []

        if circuit.id == BATCHED_PROOF_OF_WEIGHTS_MODEL_ID:
            self.score_manager.clear_proof_of_weights_queue()

        requests = []
        for uid in filtered_uids:
            synapse = self.get_synapse_request(RequestType.BENCHMARK, circuit)

            if isinstance(synapse, ProofOfWeightsSynapse):
                input_data = synapse.inputs
            else:
                input_data = synapse.query_input["public_inputs"]

            try:
                self.hash_guard.check_hash(input_data)
            except Exception as e:
                bt.logging.error(f"Hash already exists: {e}")
                safe_log({"hash_guard_error": 1})
                continue

            request = Request(
                uid=uid,
                axon=self.config.metagraph.axons[uid],
                synapse=synapse,
                circuit=circuit,
                request_type=RequestType.BENCHMARK,
                inputs=GenericInput(RequestType.RWR, input_data),
            )
            requests.append(request)

        return requests

    def select_circuit_for_benchmark(self) -> Circuit:
        """
        Select a circuit for benchmarking using weighted random selection.
        """

        circuit_id = random.choices(
            list(CIRCUIT_WEIGHTS.keys()), weights=list(CIRCUIT_WEIGHTS.values()), k=1
        )[0]

        return circuit_store.get_circuit(circuit_id)

    def format_for_query(
        self, inputs: dict[str, object], circuit: Circuit
    ) -> dict[str, object]:
        return {"public_inputs": inputs, "model_id": circuit.id}

    def get_synapse_request(
        self,
        request_type: RequestType,
        circuit: Circuit,
        request: any | None = None,
    ) -> ProofOfWeightsSynapse | QueryZkProof:
        inputs = (
            circuit.input_handler(request_type)
            if request_type == RequestType.BENCHMARK
            else circuit.input_handler(
                RequestType.RWR,
                request.inputs if hasattr(request, "inputs") else request["inputs"],
            )
        )

        if request_type == RequestType.RWR:
            if circuit.metadata.type == CircuitType.PROOF_OF_WEIGHTS:
                return ProofOfWeightsSynapse(
                    subnet_uid=circuit.metadata.netuid,
                    verification_key_hash=circuit.id,
                    proof_system=circuit.proof_system,
                    inputs=inputs.to_json(),
                    proof="",
                    public_signals="",
                )
            return QueryZkProof(
                model_id=circuit.id,
                query_input=self.format_for_query(inputs, circuit),
                query_output="",
            )

        if circuit.id in [
            SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
            BATCHED_PROOF_OF_WEIGHTS_MODEL_ID,
        ]:
            return ProofOfWeightsHandler.prepare_pow_request(
                circuit, self.score_manager.proof_of_weights_queue
            )

        if circuit.metadata.type == CircuitType.PROOF_OF_COMPUTATION:
            return QueryZkProof(
                model_id=circuit.id,
                query_input=self.format_for_query(inputs.to_json(), circuit),
                query_output="",
            )

        return ProofOfWeightsSynapse(
            subnet_uid=circuit.metadata.netuid,
            verification_key_hash=circuit.id,
            proof_system=circuit.proof_system,
            inputs=inputs.to_json(),
            proof="",
            public_signals="",
        )
