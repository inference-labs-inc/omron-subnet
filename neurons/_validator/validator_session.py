import asyncio
import json
import random
import sys
import time
import traceback

import bittensor as bt
import protocol
import torch
import wandb_logger
from _validator.reward import reward
from execution_layer.VerifiedModelSession import VerifiedModelSession
from rich.console import Console
from rich.table import Table
from utils import AutoUpdate

VALIDATOR_REQUEST_TIMEOUT_SECONDS = 30

VALIDATOR_SHOULD_QUERY_EACH_BATCH_X_TIMES_PER_EPOCH = 2
PSEUDO_SHUFFLE_EVERY_X_BLOCK = 360
EPOCH_LENGTH = 360

class ValidatorSession:
    def __init__(self, config):
        self.config = config
        self.configure()
        self.check_register()
        self.auto_update = AutoUpdate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def unpack_bt_objects(self):
        wallet = self.wallet
        metagraph = self.metagraph
        subtensor = self.subtensor
        dendrite = self.dendrite
        return wallet, metagraph, subtensor, dendrite

    def init_scores(self):
        bt.logging.info("Creating validation weights")

        try:
            self.scores = torch.load("scores.pt")
        except Exception:
            scores = torch.zeros_like(self.metagraph.S, dtype=torch.float32)

            scores = scores * torch.Tensor(
                [
                    self.metagraph.neurons[uid].axon_info.ip != "0.0.0.0"
                    for uid in self.metagraph.uids
                ]
            )
            self.scores = scores

        bt.logging.info("Successfully setup scores")

        self.log_scores()
        return self.scores

    def sync_scores_uids(self, uids):
        # If the metagraph has changed, update the weights.
        # If there are more uids than scores, add more weights.
        if len(uids) > len(self.scores):
            bt.logging.trace("Adding more weights")
            size_difference = len(uids) - len(self.scores)
            new_scores = torch.zeros(size_difference, dtype=torch.float32)
            self.scores = torch.cat((self.scores, new_scores))
            del new_scores

    def init_running_args(self):

        self.step = 0
        self.current_block = self.subtensor.block

        self.last_updated_block = self.current_block - (
            self.current_block % self.config.blocks_per_epoch
        )
        self.last_reset_weights_block = self.current_block

    def query_axons(self, requests):
        """
        Modified version of `dendrite.query` which accepts synapses unique to each axon.
        requests: list of requests
        """
        _, _, _, dendrite = self.unpack_bt_objects()

        bt.logging.trace("Querying axons")
        bt.logging.debug(f"Requests: {requests}")
        try:
            # Create a coroutine for the gather operation
            coroutine = asyncio.gather(
                *(
                    dendrite.forward(
                        axons=[request["axon"]],
                        synapse=request["synapse"],
                        timeout=VALIDATOR_REQUEST_TIMEOUT_SECONDS,
                        deserialize=False,
                    )
                    for request in requests
                )
            )

            # Run the coroutine to completion and return the result
            results = asyncio.run(coroutine)
            bt.logging.trace(f"Results: {results}")
            for i, sublist in enumerate(results):
                result = sublist[0]
                try:
                    requests[i].update(
                        {
                            "result": result,
                            "response_time": result.dendrite.process_time,
                            "deserialized": result.deserialize(),
                        }
                    )
                except Exception as e:
                    bt.logging.trace(f"Error updating request: {e}")
                    requests[i].update(
                        {
                            "result": result,
                            "response_time": sys.maxsize,
                            "deserialized": None,
                        }
                    )
            requests.sort(key=lambda x: x["uid"])
            return requests
        except Exception as e:
            bt.logging.exception("Error while querying axons. \n", e)
            return None
    
    def get_valid_validator_hotkeys(self):
        valid_hotkeys = []
        hotkeys = self.metagraph.hotkeys
        for index, hotkey in enumerate(hotkeys):
            if self.metagraph.total_stake[index] >= 1.024e3:
                valid_hotkeys.append(hotkey)
        return valid_hotkeys

    def get_validator_index(self):
        valid_hotkeys = self.get_valid_validator_hotkeys()
        try:
            return valid_hotkeys.index(self.config.wallet.hotkey)
        except ValueError:
            return -1
    
    def split_uids_in_batches(self, group_index, num_groups, queryable_uids):
        num_miners = len(queryable_uids)
        miners_per_group = num_miners // num_groups
        remaining_miners = num_miners % num_groups

        start_index = group_index * miners_per_group
        end_index = start_index + miners_per_group

        # Add remaining miners to the last group
        if group_index == num_groups - 1:
            end_index += remaining_miners

        return queryable_uids[start_index:end_index]

    def get_querable_uids(self):
        """Returns the uids of the miners that are queryable

        Returns:
            _type_: _description_
        """
        wallet, metagraph, subtensor, dendrite = self.unpack_bt_objects()
        uids = metagraph.uids.tolist()

        # Ignore validators, they're not queryable as miners.
        queryable_uids = metagraph.total_stake < 1.024e3

        # Remove the weights of miners that are not queryable.
        queryable_uids = queryable_uids * torch.Tensor(
            [metagraph.neurons[uid].axon_info.ip != "0.0.0.0" for uid in uids]
        )

        active_miners = torch.sum(queryable_uids)

        # if there are no active miners, set active_miners to 1
        if active_miners == 0:
            active_miners = 1

        # zip uids and queryable_uids, filter only the uids that are queryable, unzip, and get the uids
        zipped_uids = list(zip(uids, queryable_uids))

        filtered_uids = list(zip(*filter(lambda x: x[1], zipped_uids)))
        bt.logging.debug(f"filtered_uids: {filtered_uids}")

        if len(filtered_uids) != 0:
            filtered_uids = filtered_uids[0]

        return filtered_uids

    def update_scores(self, responses):
        """Updates scores based on the response from the miners

        Args:
            responses (_type_): [(uid, response)] array from the miners


        """
        if len(responses) == 0 or responses is None:
            return

        _, _, subtensor, _ = self.unpack_bt_objects()
        new_scores = self.scores[:]
        max_score = torch.max(self.scores)
        if max_score == 0:
            max_score = 1

        all_uids = set(range(len(self.scores)))
        response_uids = set(uid for uid, _, _, _ in responses)
        missing_uids = all_uids - response_uids

        responses.extend((uid, False, 0, 0) for uid in missing_uids)

        for uid, response, response_time, proof_size in responses:
            new_scores[uid] = reward(
                max_score, self.scores[uid], response, response_time, proof_size
            )

        if torch.sum(self.scores).item() != 0:
            self.scores = self.scores / torch.sum(self.scores)

        self.log_scores()
        self.try_store_scores()
        self.current_block = subtensor.block
        if self.current_block - self.last_updated_block > self.config.blocks_per_epoch:
            self.update_weights()

    def update_weights(self):
        wallet, metagraph, subtensor, dendrite = self.unpack_bt_objects()

        if torch.sum(self.scores).item() != 0:
            weights = self.scores / torch.sum(self.scores)
        else:
            weights = self.scores

        bt.logging.info(f"Setting weights: {weights}")

        (
            processed_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=metagraph.uids,
            weights=weights,
            netuid=self.config.netuid,
            subtensor=subtensor,
        )
        bt.logging.info(f"Processed weights: {processed_weights}")
        bt.logging.info(f"Processed uids: {processed_uids}")

        result = subtensor.set_weights(
            netuid=self.config.netuid,  # Subnet to set weights on.
            wallet=wallet,  # Wallet to sign set weights using hotkey.
            uids=processed_uids,  # Uids of the miners to set weights for.
            weights=processed_weights,  # Weights to set for the miners.
        )

        self.weights = weights
        self.log_weights()

        self.last_updated_block = metagraph.block.item()

        if result:
            bt.logging.success("✅ Successfully set weights.")
        else:
            bt.logging.error("Failed to set weights.")

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

    def verify_proof_string(self, proof_string: str, inputs):

        if proof_string == None:
            return False
        try:
            inference_session = VerifiedModelSession()
            res = inference_session.verify_proof_and_inputs(proof_string, inputs)
            inference_session.end()
            return res
        except Exception as e:
            bt.logging.error("❌ Unable to verify proof due to an error\n", e)
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
        columns = ["uid", "response_time", "proof_size", "verification_result"]
        styles = ["cyan", "magenta", "magenta", "magenta"]
        justifications = ["right"] * 4

        for col, style, justify in zip(columns, styles, justifications):
            table.add_column(col, justify=justify, style=style, no_wrap=True)

        wandb_log = {"responses": {}}
        for response in responses:
            row = [str(response[col]) for col in columns]
            table.add_row(*row)
            wandb_log["responses"][response["uid"]] = {
                col: response[col] if response[col] is not None else 0
                for col in columns[1:]
            }
            wandb_log["responses"][response["uid"]]["verification_result"] = int(
                response["verification_result"]
            )
        wandb_logger.safe_log(wandb_log)
        console.print(table)

    def deterministically_shuffle_and_batch_queryable_uids(self, metagraph, filtered_uids):
        """
        Pseudorandomly shuffles the list of queryable uids, and splits it in batches to reduce concurrent requests from multiple validators

        """
        validator_uids = metagraph.total_stake >= 1.024e3
        validator_index = self.get_validator_index()
        num_validators = len(validator_uids)

        # batch_duration_in_blocks = (EPOCH_LENGTH / VALIDATOR_SHOULD_QUERY_EACH_BATCH_X_TIMES_PER_EPOCH) // num_validators
        # batch_duration_in_blocks = (360 / 2) // 17 = 10
        # thus, if there are 17 validators, they will switch groups every 10 blocks, so every ~2 minutes

        # e.g. current_block = 2993336
        # 2993336 // batch_duration_in_blocks = 2993336 // 10 = 299333
        # (299333 + validator_index) % num_validators = (299333 + 4) % 17 = 1
        # as validator 4, I should then query the sub group 1

        batch_duration_in_blocks = (EPOCH_LENGTH / VALIDATOR_SHOULD_QUERY_EACH_BATCH_X_TIMES_PER_EPOCH) // num_validators
        batch_number_since_genesis = self.current_block // batch_duration_in_blocks
        batch_index_to_query = (batch_number_since_genesis + validator_index) % num_validators

        # We need to pseudorandomly shuffle the filtered_uids so that miner X isn't always in the same batch as miner Y
        # Which would be unfair and create batches of varying speed/quality

        # To avoid affecting the whole random package, we create an instance of it, which we seed
        seed = self.current_block // PSEUDO_SHUFFLE_EVERY_X_BLOCK
        rng = random.Random(seed)
        shuffled_filtered_uids = filtered_uids[:]
        rng.shuffle(shuffled_filtered_uids)

        # Since the shuffling was seeded, all the validators will shuffle the list the exact same way
        # without having to communicate with each other ensuring that they don't all query the same
        # miner all at once, which can happen randomly and cause deregistrations

        batched_uids = self.split_uids_in_batches(batch_index_to_query, num_validators, shuffled_filtered_uids)
        
        return batched_uids
    
    def run_step(self):
        _, metagraph, _, _ = self.unpack_bt_objects()

        # Get the uids of all miners in the network.
        uids = metagraph.uids.tolist()
        self.sync_scores_uids(uids)

        requests = []

        filtered_uids = self.get_querable_uids()

        batched_uids = self.deterministically_shuffle_and_batch_queryable_uids(metagraph, filtered_uids)

        for uid in batched_uids:
            axon = metagraph.axons[uid]
            inputs = [random.uniform(-1, 1) for _ in range(5)]
            synapse = protocol.QueryZkProof(
                query_input={
                    "model_id": [0],
                    "public_inputs": inputs,
                }
            )
            requests.append(
                {
                    "uid": uid,
                    "axon": axon,
                    "synapse": synapse,
                    "inputs": inputs,
                }
            )
        bt.logging.info(
            f"\033[92m >> Sending {len(requests)} queries for proofs to miners in the subnet \033[0m"
        )
        bt.logging.trace("Requests being sent", requests)

        try:
            responses = self.query_axons(requests)
            bt.logging.trace(f"Responses: {responses}")

            verification_results = []
            response_times = []
            proof_sizes = []
            for response in responses:
                try:

                    response["verification_result"] = self.verify_proof_string(
                        response["deserialized"], response["inputs"]
                    )
                    response["proof_size"] = len(
                        json.loads(response["deserialized"])["proof"]
                    )
                except Exception as e:
                    bt.logging.trace(
                        f"Error verifying proof or checking proof size for uid: {response['uid']}, full response: {response}, error: {e}"
                    )
                    response["proof_size"] = sys.maxsize
                    response["verification_result"] = False
                verification_results.append(response["verification_result"])
                response_times.append(response["response_time"])
                proof_sizes.append(response["proof_size"])

            self.log_responses(responses)

            self.update_scores(
                list(
                    zip(
                        filtered_uids,
                        verification_results,
                        response_times,
                        proof_sizes,
                    )
                )
            )

            self.step += 1

            # Sleep for 60s
            time.sleep(60)
        except RuntimeError as e:
            bt.logging.error(e)
            traceback.print_exc()
        except Exception as e:
            bt.logging.error(e)
            return
        # If the user interrupts the program, gracefully exit.
        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting validator.")
            exit()

    def run(self):
        bt.logging.debug("Validator started its running loop")

        wallet, metagraph, subtensor, dendrite = self.unpack_bt_objects()

        self.init_scores()
        self.init_running_args()

        while True:
            try:
                if self.config.auto_update == True:
                    self.auto_update.try_update()
                self.sync_metagraph()
                self.run_step()

            except KeyboardInterrupt:
                bt.logging.info("KeyboardInterrupt caught. Exiting validator.")
                exit()

            except Exception as e:
                bt.logging.error(
                    f"Error in validator loop \n {e} \n {traceback.format_exc()}"
                )

    def check_register(self):
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour validator: {self.wallet} is not registered to chain connection: {self.subtensor} \nRun btcli register and try again."
            )
            exit()
        else:
            # Each miner gets a unique identity (UID) in the network for differentiation.
            subnet_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(f"Running validator on uid: {subnet_uid}")
            self.subnet_uid = subnet_uid

    def configure(self):
        # === Configure Bittensor objects ====
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.sync_metagraph()
        wandb_logger.safe_init(
            "Validator",
            self.wallet,
            self.metagraph,
            self.config,
        )

    def sync_metagraph(self):
        self.metagraph.sync(subtensor=self.subtensor)
