from __future__ import annotations
from abc import ABC
import torch
import os
import time
import json
import ezkl
import bittensor as bt
from protocol import QueryZkProof, Competition
from typing import Tuple, Generator


class BaseCompetition(ABC):
    """
    Base class for competitions.
    """

    def __init__(
        self, competition_id: int, metagraph: bt.metagraph, subtensor: bt.subtensor
    ):
        self.competition_id: int = competition_id
        self.competition_directory: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            str(competition_id),
        )
        self.baseline_model: torch.nn.Module = self._load_model()
        self.metagraph: bt.metagraph = metagraph
        self.subtensor: bt.subtensor = subtensor

    def _load_model(self) -> torch.nn.Module:
        """
        Load the baseline model.
        """
        model = torch.load(os.path.join(self.competition_directory, "model.pt"))
        return model

    def sync_and_eval(self):
        """
        Sync and evaluate the circuit.
        """
        for axon, verification_key_path in self.sync_circuits():
            self.evaluate_circuit(axon, verification_key_path)

    def sync_circuits(self) -> Generator[Tuple[bt.axon, str]]:
        """
        Sync and return the circuits.
        """
        for hotkey in self.metagraph.hotkeys:
            try:
                hash = self.subtensor.get_commitment(
                    self.metagraph.netuid, self.metagraph.hotkeys.index(hotkey)
                )
                bt.logging.success(f"Syncing circuit for {hotkey} with hash {hash}")
                axon = self.metagraph.axons[self.metagraph.hotkeys.index(hotkey)]
                self.download_circuit(axon, hash)
                yield (
                    axon,
                    os.path.join(self.competition_directory, f"{hash}"),
                )
            except Exception as e:
                bt.logging.error(f"Error getting commitment for {hotkey}: {e}")
                continue

    def download_circuit(self, axon: bt.axon, hash: str):
        """
        Download the circuit.
        """
        bt.logging.info(f"Downloading circuit for hash {hash} from axon {axon}")

        out_path = os.path.join(self.competition_directory, f"{hash}")

        dendrite = bt.dendrite()

        synapse = Competition(id=self.competition_id, hash=hash)

        bt.logging.debug("Querying axon for verification key")
        response: Competition = dendrite.query(
            axons=[axon],
            synapse=synapse,
            timeout=30,
        )[0]

        if response.verification_key:
            bt.logging.info(f"Received verification key, saving to {out_path}")
            with open(out_path, "wb") as f:
                f.write(bytes.fromhex(response.verification_key))
            bt.logging.success("Successfully saved verification key")
        else:
            bt.logging.error("No verification key received from axon")
            raise ValueError(f"No verification key found for {hash}")

    def evaluate_circuit(
        self, miner_axon: bt.axon, verification_key_path: str
    ) -> float:
        """
        Evaluate a circuit and return metrics.
        """
        scores = []
        proof_sizes = []
        response_times = []
        verification_results = []

        dendrite = bt.dendrite()

        bt.logging.info(f"Evaluating circuit for {miner_axon}")

        for _ in range(10):
            test_inputs = torch.randn(4, 2)
            baseline_output = self.baseline_model(test_inputs).tolist()

            synapse = QueryZkProof(
                query_input={
                    "public_inputs": test_inputs.tolist(),
                    "model_id": self.competition_id,
                }
            )

            start_time = time.time()
            response = dendrite.query(
                axons=[miner_axon],
                synapse=synapse,
                timeout=30,
            )[0]
            response_time = time.time() - start_time
            response_times.append(response_time)

            if response.query_output:
                try:
                    output = json.loads(response.query_output)
                    proof = output["proof"]
                    public_signals = output["public_signals"]

                    proof_sizes.append(len(proof))

                    try:
                        with open("temp_proof.json", "w") as f:
                            json.dump({"proof": proof, "public": public_signals}, f)

                        verification_result = ezkl.verify(
                            "temp_proof.json",
                            os.path.join(self.competition_directory, "settings.json"),
                            verification_key_path,
                        )
                        verification_results.append(verification_result)

                        score = self.compare_outputs(baseline_output, public_signals)
                        scores.append(score)

                    except Exception as e:
                        bt.logging.error(f"Error verifying proof: {e}")
                        verification_results.append(False)
                        scores.append(0.0)

                except Exception as e:
                    bt.logging.error(f"Error parsing response: {e}")
                    scores.append(0.0)
            else:
                scores.append(0.0)

        bt.logging.info(f"Average accuracy score: {sum(scores)/len(scores)}")
        bt.logging.info(
            f"Average proof size: {sum(proof_sizes)/len(proof_sizes)} bytes"
        )
        bt.logging.info(
            f"Average response time: {sum(response_times)/len(response_times)}s"
        )
        bt.logging.info(
            f"Verification success rate: {sum(verification_results)/len(verification_results)}"
        )

        return sum(scores) / len(scores)

    def compare_outputs(self, expected: list[float], actual: list[float]) -> float:
        """
        Compare expected and actual outputs and return a score.
        """
        return (expected == actual).mean()
