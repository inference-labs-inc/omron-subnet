import base64
import json
import os

import bittensor as bt
import requests
import torch

from _validator.models.miner_response import MinerResponse

LOGGING_URL = os.getenv(
    "OMRON_LOGGING_URL",
    "https://api.omron.ai/statistics/log/",
)


def log_responses(
    metagraph: bt.metagraph,  # type: ignore
    hotkey: bt.Keypair,
    uid: int,
    responses: list[MinerResponse],
    overhead_time: float,
    block: int,
    scores: torch.Tensor,
):
    """
    Log miner responses to the centralized logging server.
    """

    data = {
        "validator_key": hotkey.ss58_address,
        "validator_uid": uid,
        "overhead_duration": overhead_time,
        "block": block,
        "responses": [response.to_log_dict(metagraph) for response in responses],
        "scores": {int(uid): float(value.item()) for uid, value in enumerate(scores)},
    }

    input_bytes = json.dumps(data).encode("utf-8")
    # sign the inputs with your hotkey
    signature = hotkey.sign(input_bytes)
    # encode the inputs and signature as base64
    signature_str = base64.b64encode(signature).decode("utf-8")

    try:
        resp = requests.post(
            LOGGING_URL,
            data=input_bytes,
            headers={
                "X-Request-Signature": signature_str,
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        bt.logging.error(f"Failed to log responses: {e}")
    return
