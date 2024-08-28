import json
import os
import traceback
from dataclasses import dataclass
import bittensor as bt
import torch

from substrateinterface import ExtrinsicReceipt
from _validator.models.completed_proof_of_weights import CompletedProofOfWeightsItem
from _validator.models.miner_response import (
    MinerResponse,
)
from _validator.utils.logging import log_pow
from constants import (
    DEFAULT_MAX_SCORE,
    DEFAULT_PROOF_SIZE,
    VALIDATOR_REQUEST_TIMEOUT_SECONDS,
)
from utils import wandb_logger

# Constants

POW_DIRECTORY = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "proof_of_weights"
)
if not os.path.exists(POW_DIRECTORY):
    os.makedirs(POW_DIRECTORY)


POW_RECEIPT_DIRECTORY = os.path.join(POW_DIRECTORY, "receipts")
if not os.path.exists(POW_RECEIPT_DIRECTORY):
    os.makedirs(POW_RECEIPT_DIRECTORY)


def to_tensor(value, dtype):
    if isinstance(value, torch.Tensor):
        return value.clone().detach().to(dtype)
    return torch.tensor(value, dtype=dtype)


dummy_miner_response = MinerResponse.empty()


@dataclass
class ProofOfWeightsItem:
    max_score: torch.Tensor
    previous_score: torch.Tensor
    verification_result: torch.Tensor
    proof_size: torch.Tensor
    response_time: torch.Tensor
    median_max_response_time: torch.Tensor
    min_response_time: torch.Tensor
    block_number: torch.Tensor
    validator_uid: torch.Tensor
    uid: torch.Tensor

    def __post_init__(self):
        self.max_score = to_tensor(self.max_score, torch.float32)
        self.previous_score = to_tensor(self.previous_score, torch.float32)
        self.verification_result = to_tensor(self.verification_result, torch.bool)
        self.proof_size = to_tensor(self.proof_size, torch.int64)
        self.response_time = to_tensor(self.response_time, torch.float32)
        self.median_max_response_time = to_tensor(
            self.median_max_response_time, torch.float32
        )
        self.min_response_time = to_tensor(self.min_response_time, torch.float32)
        self.block_number = to_tensor(self.block_number, torch.int64)
        self.validator_uid = to_tensor(self.validator_uid, torch.int64)
        self.uid = to_tensor(self.uid, torch.int64)

    @staticmethod
    def from_miner_response(
        response: MinerResponse,
        max_score,
        previous_score,
        median_max_response_time,
        min_response_time,
        block_number,
        validator_uid,
    ):
        return ProofOfWeightsItem(
            max_score=max_score,
            previous_score=previous_score,
            verification_result=torch.tensor(
                response.verification_result, dtype=torch.bool
            ),
            proof_size=torch.tensor(response.proof_size, dtype=torch.int64),
            response_time=(
                torch.tensor(response.response_time, dtype=torch.float32)
                if response.verification_result
                else median_max_response_time
            ),
            median_max_response_time=median_max_response_time,
            min_response_time=min_response_time,
            block_number=block_number,
            validator_uid=validator_uid,
            uid=torch.tensor(response.uid, dtype=torch.int64),
        )

    @staticmethod
    def pad_items(
        items: list["ProofOfWeightsItem"], target_item_count: int = 256
    ) -> list["ProofOfWeightsItem"]:
        if len(items) == 0:
            return [ProofOfWeightsItem.empty()] * target_item_count

        # Pad or truncate the input list to exactly target_item_count
        if len(items) < target_item_count:
            items.extend(
                [ProofOfWeightsItem.empty()] * (target_item_count - len(items))
            )
        elif len(items) > target_item_count:
            items = items[:target_item_count]

        return items

    @staticmethod
    def empty():
        return ProofOfWeightsItem(
            max_score=torch.tensor(DEFAULT_MAX_SCORE, dtype=torch.float32),
            previous_score=torch.tensor(0, dtype=torch.float32),
            verification_result=torch.tensor(0, dtype=torch.int64),
            proof_size=torch.tensor(DEFAULT_PROOF_SIZE, dtype=torch.int64),
            response_time=torch.tensor(
                VALIDATOR_REQUEST_TIMEOUT_SECONDS, dtype=torch.float32
            ),
            median_max_response_time=torch.tensor(
                VALIDATOR_REQUEST_TIMEOUT_SECONDS, dtype=torch.float32
            ),
            min_response_time=torch.tensor(0, dtype=torch.float32),
            block_number=torch.tensor(0, dtype=torch.int64),
            validator_uid=torch.tensor(0, dtype=torch.int64),
            uid=torch.tensor(0, dtype=torch.int64),
        )

    @staticmethod
    def from_tensor(tensor: torch.Tensor):
        return ProofOfWeightsItem(
            max_score=tensor[0],
            previous_score=tensor[1],
            verification_result=tensor[2],
            proof_size=tensor[3],
            response_time=tensor[4],
            median_max_response_time=tensor[5],
            min_response_time=tensor[6],
            block_number=tensor[7],
            validator_uid=tensor[8],
            uid=tensor[9],
        )

    def to_tensor(self):
        return torch.tensor(
            [
                self.max_score,
                self.previous_score,
                self.verification_result,
                self.proof_size,
                self.response_time,
                self.median_max_response_time,
                self.min_response_time,
                self.block_number,
                self.validator_uid,
                self.uid,
            ]
        )

    @staticmethod
    def merge_items(
        *items_lists: list["ProofOfWeightsItem"],
    ) -> list["ProofOfWeightsItem"]:
        return [item for items in items_lists for item in items]

    @staticmethod
    def to_dict_list(items: list["ProofOfWeightsItem"]):
        result = {
            "max_score": [],
            "previous_score": [],
            "verification_result": [],
            "proof_size": [],
            "response_time": [],
            "median_max_response_time": [],
            "min_response_time": [],
            "block_number": [],
            "validator_uid": [],
            "uid": [],
        }
        for item in items:
            result["max_score"].append(item.max_score.item())
            result["previous_score"].append(item.previous_score.item())
            result["verification_result"].append(item.verification_result.item())
            result["proof_size"].append(item.proof_size.item())
            result["response_time"].append(item.response_time.item())
            result["median_max_response_time"].append(
                item.median_max_response_time.item()
            )
            result["min_response_time"].append(item.min_response_time.item())
            result["block_number"].append(item.block_number.item())
            result["validator_uid"].append(item.validator_uid.item())
            result["uid"].append(item.uid.item())
        return result

    @classmethod
    def from_dict_list(cls, data: dict) -> list["ProofOfWeightsItem"]:
        items = []
        for i in range(len(data["max_score"])):
            items.append(
                cls(
                    max_score=data["max_score"][i],
                    previous_score=data["previous_score"][i],
                    verification_result=data["verification_result"][i],
                    proof_size=data["proof_size"][i],
                    response_time=data["response_time"][i],
                    median_max_response_time=data["median_max_response_time"][i],
                    min_response_time=data["min_response_time"][i],
                    block_number=data["block_number"][i],
                    validator_uid=data["validator_uid"][i],
                    uid=data["uid"][i],
                )
            )
        return items


def save_proof_of_weights(public_signals: list, proof: str):
    """
    Save the proof of weights to a JSON file.

    Args:
        public_signals (list): The public signals data as a JSON array.
        proof (str): The proof.

    This function saves the proof of weights as a JSON file and logs them in WandB.
    """
    try:
        log_pow({"public_signals": public_signals, "proof": proof})
    except Exception as e:
        bt.logging.error(f"Error logging proof of weights: {e}")
        traceback.print_exc()
    try:
        block_numbers_out = public_signals[1025:2049]

        unique_block_numbers = list(set(block_numbers_out))
        unique_block_numbers.sort()
        block_number = "_".join(str(num) for num in unique_block_numbers)
        file_path = os.path.join(POW_DIRECTORY, f"{block_number}.json")

        proof_json = {"public_signals": public_signals, "proof": proof}

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(proof_json, f)
        bt.logging.success(f"Proof of weights saved to {file_path}")
    except Exception as e:
        bt.logging.error(f"Error saving proof of weights to file: {e}")
        traceback.print_exc()


def log_and_commit_proof(
    hotkey: bt.Keypair,
    subtensor: bt.subtensor,
    proof_list: list[CompletedProofOfWeightsItem],
):
    """
    Log and commit aggregated proofs on-chain for the previous weight setting period.

    Args:
        hotkey (bt.Keypair): Keypair for signing the transaction.
        subtensor (bt.subtensor): Subtensor instance for blockchain interaction.
        proof_list (list[CompletedProofOfWeightsItem]): List of proofs to be included in the transaction.

    This function commits proofs of weights to the chain as a system remark, making PoW immutable.
    """
    if len(proof_list) == 0:
        bt.logging.warning("No proofs to log to the chain.")
        wandb_logger.safe_log({"on_chain_proof_of_weights_committed": False})
        return
    bt.logging.debug(
        f"Logging and committing proof(s) of weights. Proof count: {len(proof_list)}"
    )

    remark_bodies = []
    for item in proof_list:
        remark_bodies.append(item.to_remark())
    bt.logging.trace(f"Remark bodies: {remark_bodies}")
    try:
        bt.logging.debug("Committing on-chain proof of weights aggregation")
        with subtensor.substrate as substrate:
            remark_call = substrate.compose_call(
                call_module="System",
                call_function="remark",
                call_params={"remark": json.dumps(remark_bodies)},
            )
            extrinsic = substrate.create_signed_extrinsic(
                call=remark_call, keypair=hotkey
            )
            result = substrate.submit_extrinsic(
                extrinsic, wait_for_inclusion=True, wait_for_finalization=True
            )
            result.process_events()
            bt.logging.info(
                f"On-chain proof of weights committed. Hash: {result.extrinsic_hash}"
            )
            wandb_logger.safe_log(
                {
                    "on_chain_proof_of_weights_committed": True,
                    "pow_cost": result.total_fee_amount,
                }
            )
            save_receipt(result)
    except Exception as e:
        bt.logging.error(
            f"""\033[31m
Error committing on-chain proof of weights using hotkey: {hotkey.ss58_address}.
Validators within the Omron subnet are compensated by the subnet for providing on-chain proof of weights.
Please reach out via Discord to request TAO for proof of weights, which we'll provide to your hotkey.
\033[0m{e}
            """
        )
        bt.logging.trace(traceback.format_exc())


def save_receipt(receipt: ExtrinsicReceipt):
    with open(
        os.path.join(POW_RECEIPT_DIRECTORY, f"{receipt.extrinsic_hash}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump({k: str(v) for k, v in receipt.__dict__.items()}, f, indent=4)
