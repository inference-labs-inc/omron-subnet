from _validator.pow.proof_of_weights_handler import ProofOfWeightsHandler
import bittensor as bt
import random
from protocol import ProofOfWeightsSynapse, QueryZkProof

from _validator.scoring.score_manager import ScoreManager
from _validator.core.api import ValidatorAPI
from _validator.config import ValidatorConfig
from constants import (
    BATCHED_PROOF_OF_WEIGHTS_MODEL_ID,
    DEFAULT_NETUID,
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
        Prepare a batch of requests for the provided UIDs.

        Args:
            filtered_uids (list): List of filtered UIDs to query.

        Returns:
            list: List of prepared requests.
        """

        request_type = (
            RequestType.BENCHMARK
            if not self.api.external_requests_queue
            else RequestType.RWR
        )

        netuid = self.config.subnet_uid
        circuit = self.select_circuit_for_benchmark()
        request = None
        if request_type == RequestType.RWR:
            netuid, request = self.api.external_requests_queue.pop()
            bt.logging.debug(f"Processing external request for netuid {netuid}")

            target_netuid = (
                DEFAULT_NETUID if netuid == self.config.subnet_uid else netuid
            )
            circuit = circuit_store.get_latest_circuit_for_netuid(target_netuid)

        bt.logging.info(
            f"The next round of requests will be for {circuit} in {request_type} mode"
        )

        requests = [
            Request(
                uid=uid,
                axon=self.config.metagraph.axons[uid],
                synapse=self.get_synapse_request(uid, request_type, circuit, request),
                circuit=circuit,
                request_type=request_type,
            )
            for uid in filtered_uids
        ]

        for request in requests:
            input_data = (
                request.synapse.inputs
                if request.circuit.metadata.type == CircuitType.PROOF_OF_WEIGHTS
                else request.synapse.query_input["public_inputs"]
            )
            request.inputs = GenericInput(RequestType.RWR, input_data)
            try:
                self.hash_guard.check_hash(input_data)
            except Exception as e:
                bt.logging.error(f"Hash already exists: {e}")
                safe_log({"hash_guard_error": 1})
                continue

        if circuit.id == BATCHED_PROOF_OF_WEIGHTS_MODEL_ID:
            self.score_manager.clear_proof_of_weights_queue()

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
        uid: int,
        request_type: RequestType,
        circuit: Circuit,
        request: dict[str, object] | None = None,
    ) -> ProofOfWeightsSynapse | QueryZkProof:
        inputs = (
            circuit.input_handler(request_type)
            if request_type == RequestType.BENCHMARK
            else circuit.input_handler(RequestType.RWR, request["inputs"])
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
            # We'll forward the responsibility of handling these to the internal proof of weights handler
            return ProofOfWeightsHandler.prepare_pow_request(
                circuit, self.score_manager.proof_of_weights_queue
            )

        # Otherwise, we'll prepare a regular benchmark request depending on the circuit type
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
