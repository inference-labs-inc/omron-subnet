from __future__ import annotations

import asyncio
import json
import os
import random
import secrets
import sys
import time
import traceback
import uuid
from collections.abc import Generator, Iterable
from enum import Enum
from typing import NoReturn, Union

import bittensor as bt
import ezkl
import protocol
import torch
import wandb_logger
from _validator.reward import Reward
from execution_layer.VerifiedModelSession import VerifiedModelSession
from rich.console import Console
from rich.table import Table
from utils import AutoUpdate, clean_temp_files, hotkey_to_split_tensor

# Hash of the reward model's VK
PROOF_OF_WEIGHTS_MODEL_ID: str = (
    "0a92bc32ea02abe54159da70aeb541d52c3cba27c8708669eda634e096a86f8b"
)
# The maximum timespan allowed for miners to respond to a query
VALIDATOR_REQUEST_TIMEOUT_SECONDS = 60
# The timeout for aggregation requests
VALIDATOR_AGG_REQUEST_TIMEOUT_SECONDS = 600
# Maximum number of concurrent requests that the validator will handle
MAX_CONCURRENT_REQUESTS = 16
# Default proof size when we're unable to determine the actual size
DEFAULT_PROOF_SIZE = 5000
# Enables aggregation requests for proof of weights
ENABLE_POW_AGGREGATION = False


class ProofOfWeightsStatus(Enum):
    """
    Status of proof of weights requests
    """

    queued = "queued"
    proven = "proven"
    failed_to_prove = "failed_to_prove"
    aggregating = "aggregating"
    aggregated = "aggregated"
    failed_to_aggregate = "failed_to_aggregate"


class ProofOfWeightsItem:
    """
    Proof of weights object
    """

    def __init__(self, inputs, uid):
        self.inputs = inputs
        self.uid = uid
        self.status = "queued"

    def update_status(self, status):
        self.status = status


class ValidatorSession:

    def __init__(self, config):
        self.config = config
        self.configure()
        self.check_register()
        self.auto_update = AutoUpdate()
        self.scores = None
        self.weights = None
        self.current_block = None
        self.step = 0
        self.last_updated_block = None
        self.proof_of_weights_queue = []
        self.pow_directory = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "proof_of_weights"
        )
        self.pow_aggregation_queue = [[] for _ in range(256)]
        if not os.path.exists(self.pow_directory):
            os.makedirs(self.pow_directory)
        self.aggregation_active = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def init_scores(self):
        bt.logging.info("Creating validation weights")

        try:
            self.scores = torch.load("scores.pt")
        except Exception:
            scores = torch.zeros_like(self.metagraph.S, dtype=torch.float32)

            self.scores = scores * torch.Tensor(
                [
                    # trunk-ignore(bandit/B104)
                    self.metagraph.neurons[uid].axon_info.ip != "0.0.0.0"
                    for uid in self.metagraph.uids
                ]
            )

        bt.logging.info("Successfully setup scores")

        self.log_scores()
        return self.scores

    def sync_scores_uids(self, uids):
        """
        If the metagraph has changed, update the weights.
        If there are more uids than scores, add more weights.
        """
        if len(uids) > len(self.scores):
            bt.logging.trace("Adding more weights")
            size_difference = len(uids) - len(self.scores)
            new_scores = torch.zeros(size_difference, dtype=torch.float32)
            self.scores = torch.cat((self.scores, new_scores))
            del new_scores

    async def query_axons(self, requests: list[dict]) -> list[dict]:
        """
        Modified version of `dendrite.query` which accepts synapses unique to each axon.
        requests: list of requests
        """
        bt.logging.trace("Querying axons")
        random.shuffle(requests)
        bt.logging.debug(f"Shuffled requests: {requests}")
        # Create a semaphore that locks the number of concurrent requests to MAX_CONCURRENT_REQUESTS
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def send_request(request):
            async with semaphore:
                return await self.dendrite.forward(
                    axons=[request["axon"]],
                    synapse=request["synapse"],
                    timeout=(
                        VALIDATOR_REQUEST_TIMEOUT_SECONDS
                        if not request["aggregation"]
                        else VALIDATOR_AGG_REQUEST_TIMEOUT_SECONDS
                    ),
                    deserialize=False,
                )

        tasks = [send_request(request) for request in requests]

        try:
            # Create a coroutine for the gather operation
            results = await asyncio.gather(*tasks)
            bt.logging.trace(f"Results: {results}")
            for i, sublist in enumerate(results):
                result = sublist[0]
                try:
                    requests[i].update(
                        {
                            "result": result,
                            "response_time": (
                                result.dendrite.process_time
                                if result.dendrite.process_time is not None
                                else (
                                    VALIDATOR_REQUEST_TIMEOUT_SECONDS
                                    if not requests[i]["aggregation"]
                                    else VALIDATOR_AGG_REQUEST_TIMEOUT_SECONDS
                                )
                            ),
                            "deserialized": result.deserialize(),
                        }
                    )
                except Exception as e:
                    bt.logging.trace(f"Error updating request: {e}")
                    traceback.print_exc()
                    requests[i].update(
                        {
                            "result": result,
                            "response_time": (
                                VALIDATOR_REQUEST_TIMEOUT_SECONDS
                                if not requests[i]["aggregation"]
                                else VALIDATOR_AGG_REQUEST_TIMEOUT_SECONDS
                            ),
                            "deserialized": None,
                        }
                    )
            requests.sort(key=lambda x: x["uid"])
            return requests
        except Exception as e:
            bt.logging.exception("Error while querying axons. \n", e)
            traceback.print_exc()
            return None

    def get_queryable_uids(self, uids: list[int]) -> Generator[int, None, None]:
        """
        Returns the uids of the miners that are queryable
        """
        # Ignore validators, they're not queryable as miners (torch.nn.Parameter)
        queryable_flags: Iterable[bool] = self.metagraph.total_stake < 1.024e3

        for uid, is_queryable in zip(uids, queryable_flags):

            # trunk-ignore(bandit/B104)
            if self.metagraph.neurons[uid].axon_info.ip != "0.0.0.0" and is_queryable:
                yield uid

    def update_scores(
        self, responses: list[tuple[int, bool, float, int, list[str], dict]]
    ) -> None:
        """
        Updates scores based on the response from the miners
        Args:
            responses (list): `[(uid, verification_result, response_time, proof_size)]`
        """
        if not responses:
            bt.logging.error("No responses received, skipping score update")
            return
        max_score = torch.max(self.scores)
        if max_score == 0:
            max_score = 1

        # add response info for non-queryable uids with zeros (those are validators' uids)
        all_uids = set(range(len(self.scores)))
        response_uids = set(r[0] for r in responses)
        missing_uids = all_uids - response_uids

        responses.extend((uid, False, 0, 0, [0], {}) for uid in missing_uids)
        max_response_time = max(
            (response[2] if response[2] is not None else 0 for response in responses),
            default=0,
        )

        bt.logging.info("Responses: ", responses)
        reward_model = Reward()

        for response in responses:
            try:
                if not isinstance(response, tuple) or len(response) != 6:
                    bt.logging.error(f"Invalid response format: {response}")
                    continue
                uid, verified, response_time, proof_size, model_id, proof_json = (
                    response
                )

                if not isinstance(uid, int):
                    bt.logging.error(f"Invalid uid format: {uid}, skipping")
                    continue

                median_max_response_time = torch.median(
                    torch.tensor(
                        sorted(
                            [
                                (
                                    response[2]
                                    if response[2] is not None
                                    else VALIDATOR_REQUEST_TIMEOUT_SECONDS
                                )
                                for response in responses
                            ]
                        )[-max(int(len(responses) * 0.05), 1) :]
                    )
                )

                min_response_time = torch.min(
                    (
                        torch.tensor(
                            [
                                (
                                    response[2]
                                    if response[2] is not None
                                    else 0
                                )
                                for response in responses
                            ]
                        )
                    )
                )

                if model_id == [PROOF_OF_WEIGHTS_MODEL_ID]:
                    bt.logging.info(f"Received proof of weights for UID {uid}")

                    torch_arguments = [
                        max_score,
                        self.scores[uid],
                        torch.tensor(verified),
                        torch.tensor(proof_size),
                        torch.tensor(response_time),
                        median_max_response_time,
                        min_response_time,
                        hotkey_to_split_tensor(self.wallet.hotkey.public_key.hex()),
                        self.metagraph.block,
                        uid,
                    ]

                    bt.logging.debug(
                        f"Calculating score for miner given the following inputs: "
                        f"{torch_arguments}"
                    )

                    output_tensor = reward_model.forward(*torch_arguments)

                    bt.logging.trace(f"Reward output tensor: {output_tensor}")
                    self.scores[uid] = output_tensor[0]
                    bt.logging.debug(f"Updated score for UID {uid}: {self.scores[uid]}")

                if model_id != PROOF_OF_WEIGHTS_MODEL_ID:
                    # If the model is not the SN2 proof of weights model itself, we send the proof with inputs into the
                    # proof of weights queue
                    bt.logging.debug(
                        f"Appending proof of weights for UID {uid} to queue"
                    )
                    proof_of_weights = ProofOfWeightsItem(
                        [
                            [float(max_score)],
                            [float(self.scores[uid])],
                            [verified],
                            [proof_size],
                            [float(torch.tensor(response_time))],
                            [float(max_response_time)],
                            [float(min_response_time)],
                            hotkey_to_split_tensor(
                                self.wallet.hotkey.public_key.hex()
                            ).tolist(),
                            [self.metagraph.block.item()],
                            [uid],
                        ],
                        uid,
                    )
                    self.proof_of_weights_queue.append(proof_of_weights)

                bt.logging.trace(f"Calculated score for UID {uid}: {self.scores[uid]}")
            except Exception as e:
                bt.logging.error(f"Error calculating score for uid: {uid}, error: {e}")
                traceback.print_exc()
        self.log_scores()
        self.try_store_scores()
        self.current_block = self.subtensor.block
        if self.current_block - self.last_updated_block > self.config.blocks_per_epoch:
            self.update_weights(torch.tensor(list(all_uids)))
        self.aggregation_active = True

    def update_weights(self, uids: torch.Tensor) -> None:
        if uids.shape[0] == 0 or len(self.scores) == 0:
            bt.logging.warning("No uids or scores to update weights. Skipping.")
            return

        if torch.sum(self.scores).item() != 0:
            weights = self.scores / torch.sum(self.scores)
        else:
            weights = self.scores

        bt.logging.info(f"Setting weights: {weights}")

        (
            processed_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=uids,
            weights=weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
        )
        bt.logging.info(f"Processed weights: {processed_weights}")
        bt.logging.info(f"Processed uids: {processed_uids}")

        is_weights_set, set_weights_msg = self.subtensor.set_weights(
            netuid=self.config.netuid,  # Subnet to set weights on.
            wallet=self.wallet,  # Wallet to sign set weights using hotkey.
            uids=processed_uids,  # Uids of the miners to set weights for.
            weights=processed_weights,  # Weights to set for the miners.
        )

        if not is_weights_set:
            bt.logging.error(f"Failed to set weights - {set_weights_msg}.")
            return

        bt.logging.success(f"✅ Successfully set weights - {set_weights_msg}")
        self.weights = weights
        self.last_updated_block = self.metagraph.block.item()
        self.log_weights()

    def log_scores(self):
        table = Table(title="scores")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("score", justify="right", style="magenta", no_wrap=True)
        log_data = {"scores": {}}
        for uid, score in enumerate(self.scores):
            log_data["scores"][uid] = score
            table.add_row(str(uid), str(round(score.item(), 4)))
        wandb_logger.safe_log(log_data)
        console = Console()
        console.print(table)

    def try_store_scores(self):
        try:
            torch.save(self.scores, "scores.pt")
        except Exception as e:
            bt.logging.info(f"Error at storing scores {e}")

    def log_verify_result(self, results):
        table = Table(title="proof verification result")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("Verified?", justify="right", style="magenta", no_wrap=True)
        verification_results = {"verification_results": {}}
        for uid, result in results:
            verification_results["verification_results"][uid] = int(result)
            table.add_row(str(uid), str(result))
        wandb_logger.safe_log(verification_results)
        console = Console()
        console.print(table)

    def log_weights(self):
        table = Table(title="weights")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", justify="right", style="magenta", no_wrap=True)
        weights = {"weights": {}}
        for uid, score in enumerate(self.weights):
            weights["weights"][uid] = score
            table.add_row(str(uid), str(round(score.item(), 4)))
        wandb_logger.safe_log(weights)

        console = Console()
        console.print(table)

    def verify_proof_string(
        self, proof_string: str, inputs: list[float], model_id: list[str or int]
    ) -> bool:
        if not proof_string:
            return False
        try:
            if isinstance(model_id, list) and model_id[0] == 0:
                inputs = [inputs]
            inference_session = VerifiedModelSession(
                public_inputs=inputs, model_id=model_id
            )
            res = inference_session.verify_proof_and_inputs(proof_string, inputs)
            inference_session.end()
            return res
        except Exception as e:
            bt.logging.error("❌ Unable to verify proof due to an error\n", e)
            traceback.print_exc()
            bt.logging.trace(
                f"Offending proof string: {proof_string}\n Inputs: {inputs}"
            )

        return False

    def log_responses(self, responses):
        """
        Log response information to the console and to wandb
        """
        console = Console()
        table = Table(title="responses")
        columns = [
            "uid",
            "verification_result",
            "response_time",
            "proof_size",
            "model_id",
        ]
        styles = ["cyan", "magenta", "magenta", "magenta", "magenta"]
        justifications = ["right"] * 5

        for col, style, justify in zip(columns, styles, justifications):
            table.add_column(col, justify=justify, style=style, no_wrap=True)

        wandb_log = {"responses": {}}
        for response in responses:
            row = [str(response[index]) for index in range(len(columns))]
            table.add_row(*row)
            wandb_log["responses"][response[0]] = {
                columns[index]: response[index] if response[index] is not None else 0
                for index in range(1, len(columns))
            }
            wandb_log["responses"][response[0]]["verification_result"] = int(
                response[3]
            )
        wandb_logger.safe_log(wandb_log)
        console.print(table)

    def prepare_requests(self, uids: list[int]) -> list[dict]:
        requests = []
        use_aggregation = (
            len(self.pow_aggregation_queue) >= len(uids)
            and self.aggregation_active
            and ENABLE_POW_AGGREGATION
        )
        if self.aggregation_active:
            self.aggregation_active = False
        use_proof_of_weights = len(self.proof_of_weights_queue) >= len(uids)

        for uid in uids:
            model_id: Union[str, list[Union[str, int]]] = [0]
            inputs: list[float] = [
                secrets.SystemRandom().uniform(-1, 1) for _ in range(5)
            ]
            axon = self.metagraph.axons[uid]
            # If there are enough proof of weights requests then replace regular requests with proof of weights requests
            if use_aggregation:
                bt.logging.info(
                    f"Preparing request for proof of weights aggregation: {uid}"
                )
                file_paths = self.pow_aggregation_queue.pop(0)
                inputs = []
                for file_path in file_paths:
                    with open(file_path, "r", encoding="utf-8") as file:
                        inputs.append(file.read())
                model_id = PROOF_OF_WEIGHTS_MODEL_ID
                bt.logging.trace(
                    f"Preparing request for proof of weights aggregation: {uid}, inputs: {inputs}, model_id: {model_id}"
                )
                synapse = protocol.QueryForProofAggregation(
                    proofs=inputs,
                    model_id=model_id,
                )
                requests.append(
                    {
                        "uid": uid,
                        "axon": axon,
                        "synapse": synapse,
                        "inputs": inputs,
                        "model_id": model_id,
                        "aggregation": True,
                    }
                )
                continue

            if use_proof_of_weights:
                inputs = self.proof_of_weights_queue.pop(0).inputs
                model_id = [PROOF_OF_WEIGHTS_MODEL_ID]
                bt.logging.trace(
                    f"Preparing request for proof of weights uid: {uid}, inputs: {inputs}, model_id: {model_id}"
                )
            synapse = protocol.QueryZkProof(
                query_input={
                    "model_id": model_id,
                    "public_inputs": inputs,
                }
            )
            requests.append(
                {
                    "uid": uid,
                    "axon": axon,
                    "synapse": synapse,
                    "inputs": inputs,
                    "model_id": model_id,
                    "aggregation": False,
                }
            )
        return requests

    def save_proof_of_weights(self, proof_json: dict):
        file_path = os.path.join(self.pow_directory, f"{uuid.uuid4()}.json")
        try:
            instances = proof_json["instances"][0]
            validator_hotkey = "".join(
                [
                    chr(ezkl.felt_to_int(char))
                    for char in instances[-66:-2]
                ]
            )
            block_number = ezkl.felt_to_int(instances[-2])
            miner_uid = ezkl.felt_to_int(instances[-1])
            file_path = os.path.join(
                self.pow_directory,
                f"{block_number}_{validator_hotkey}_{miner_uid}.json",
            )
            self.pow_aggregation_queue[miner_uid].append(file_path)
        except Exception as e:
            bt.logging.error(
                f"Error extracting instance values to generate descriptive filename: {e}"
            )
            traceback.print_exc()
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(proof_json, f)
        except Exception as e:
            bt.logging.error(f"Error saving proof of weights to file: {e}")
            traceback.print_exc()

    def process_single_response(self, response: dict) -> tuple:
        """
        Process a single response, verifying the proof and extracting relevant fields.
        """
        try:
            proof_raw = []
            verification_result = self.verify_proof_string(
                response["deserialized"],
                response["inputs"],
                model_id=response["model_id"],
            )
            try:
                proof_json = json.loads(response["deserialized"])
                proof_raw = proof_json["proof"]
            except Exception as e:
                bt.logging.debug(
                    f"Unable to parse proof json for response: {response}, error: {e}"
                )
                proof_json = {}

            proof_size = DEFAULT_PROOF_SIZE
            try:
                proof_size = len(proof_raw)
            except Exception as e:
                bt.logging.debug(
                    f"Unable to determine proof size for response: {response}, error: {e}"
                )
            response["proof_size"] = proof_size

            if (
                response["model_id"][0] == PROOF_OF_WEIGHTS_MODEL_ID
                and verification_result
            ):
                self.save_proof_of_weights(proof_json)

            return (
                response["uid"],
                verification_result,
                response["response_time"],
                response["proof_size"],
                response["model_id"],
                proof_json,
            )

        except json.JSONDecodeError:
            bt.logging.error(f"JSON decoding failed for response: {response}")
            return response["uid"], False, VALIDATOR_REQUEST_TIMEOUT_SECONDS, 0, [0], {}
        except Exception as e:
            bt.logging.error(f"Error processing response: {response}, error: {e}")
            return response["uid"], False, VALIDATOR_REQUEST_TIMEOUT_SECONDS, 0, [0], {}

    def log_and_commit_proof(
        self, response: dict, instances: list[str], proof_raw: list[int]
    ):
        """
        Commit aggregated proofs on-chain for the previous weight setting period.
        """
        bt.logging.debug(
            f"Logging and committing proof of weights. Instances: {instances}, proof: {proof_raw}"
        )
        remark_body = {
            "type": "proof_of_weights",
            "model_id": response["model_id"],
            "instances": instances,
            "proof": proof_raw,
            "completed_by_uid": response["uid"],
        }
        bt.logging.trace(f"Remark body: {remark_body}")

        try:
            bt.logging.debug("Committing on-chain proof of weights aggregation")

            with self.subtensor.substrate as substrate:
                remark_call = substrate.compose_call(
                    call_module="System",
                    call_function="remark",
                    call_params={"remark": remark_body},
                )
                extrinsic = substrate.create_signed_extrinsic(
                    call=remark_call, keypair=self.wallet.hotkey
                )
                result = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
                result.process_events()
                bt.logging.info(
                    f"On-chain proof of weights committed. Hash: {result.extrinsic_hash}"
                )
        except Exception as e:
            bt.logging.error(
                f"Error committing on-chain proof of weights for uid {response['uid']}: {e}"
            )
            traceback.print_exc()

    def run_step(self) -> None:
        """
        Run a single step of the validator.
        """

        uids = self.metagraph.uids.tolist()
        self.sync_scores_uids(uids)

        filtered_uids = list(self.get_queryable_uids(uids))
        random.shuffle(filtered_uids)

        requests = self.prepare_requests(filtered_uids)

        bt.logging.info(
            f"\033[92m >> Sending {len(requests)} queries for proofs to miners in the subnet \033[0m"
        )
        bt.logging.trace("Requests being sent", requests)

        try:
            loop = asyncio.get_event_loop()
            responses = loop.run_until_complete(self.query_axons(requests))
            bt.logging.trace(f"Responses: {responses}")
            processed_responses = [self.process_single_response(r) for r in responses]

            self.update_scores(processed_responses)

            self.log_responses(processed_responses)

            # Sleep for 60s
            time.sleep(60)
        except RuntimeError as e:
            bt.logging.error(
                f"A runtime error occurred in the main validator loop\n{e}"
            )
            traceback.print_exc()
        except Exception as e:
            bt.logging.error(f"An error occurred in the main validator loop\n{e}")
            traceback.print_exc()
            return
        # If the user interrupts the program, gracefully exit.
        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting validator.")
            clean_temp_files()
            sys.exit(0)

    def run(self) -> NoReturn:
        """
        Start the validator session and run the main loop
        """
        bt.logging.debug("Validator started its running loop")
        self.init_scores()
        self.current_block = self.subtensor.block
        # Set the last updated block to the last epoch boundary
        self.last_updated_block = self.current_block - (
            self.current_block % self.config.blocks_per_epoch
        )

        while True:
            try:
                if not self.config.no_auto_update:
                    self.auto_update.try_update()
                else:
                    bt.logging.info(
                        "Automatic updates are disabled, skipping version check"
                    )
                self.metagraph.sync(subtensor=self.subtensor)
                self.run_step()

            except KeyboardInterrupt:
                bt.logging.info("KeyboardInterrupt caught. Exiting validator.")
                clean_temp_files()
                sys.exit(0)

            except Exception as e:
                bt.logging.error(
                    f"Error in validator loop \n {e} \n {traceback.format_exc()}"
                )

    def check_register(self):
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour validator: {self.wallet} is not registered to the chain: "
                f"{self.subtensor} \nRun btcli register and try again."
            )
            sys.exit(1)
        else:
            # Each miner gets a unique identity (UID) in the network for differentiation.
            subnet_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(f"Running validator on uid: {subnet_uid}")
            self.subnet_uid = subnet_uid

    def configure(self):
        """
        Configure Bittensor objects
        """
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)

        wandb_logger.safe_init(
            "Validator",
            self.wallet,
            self.metagraph,
            self.config,
        )
