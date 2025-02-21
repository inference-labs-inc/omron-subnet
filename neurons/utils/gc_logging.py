import base64
import json
import os
from typing import Optional

import bittensor as bt
import requests
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from _validator.models.miner_response import MinerResponse

LOGGING_URL = os.getenv(
    "LOGGING_URL",
    "https://api.omron.ai/statistics/log/",
)

COMPETITION_LOGGING_URL = os.getenv(
    "COMPETITION_LOGGING_URL",
    "https://api.omron.ai/statistics/competition/log/",
)

session = requests.Session()
retries = Retry(total=3, backoff_factor=0.1)
session.mount("https://", HTTPAdapter(max_retries=retries))


def log_responses(
    metagraph: bt.metagraph,
    hotkey: bt.Keypair,
    uid: int,
    responses: list[MinerResponse],
    overhead_time: float,
    block: int,
    scores: torch.Tensor,
) -> Optional[requests.Response]:
    """
    Log miner responses to the centralized logging server.
    """

    data = {
        "validator_key": hotkey.ss58_address,
        "validator_uid": uid,
        "overhead_duration": overhead_time,
        "block": block,
        "responses": [response.to_log_dict(metagraph) for response in responses],
        "scores": {k: float(v.item()) for k, v in enumerate(scores) if v.item() > 0},
    }

    input_bytes = json.dumps(data).encode("utf-8")
    # sign the inputs with your hotkey
    signature = hotkey.sign(input_bytes)
    # encode the inputs and signature as base64
    signature_str = base64.b64encode(signature).decode("utf-8")

    try:
        return session.post(
            LOGGING_URL,
            data=input_bytes,
            headers={
                "X-Request-Signature": signature_str,
                "Content-Type": "application/json",
            },
            timeout=5,
        )
    except requests.exceptions.RequestException as e:
        bt.logging.error(f"Failed to log responses: {e}")
        return None


def gc_log_competition_metrics(
    metrics: dict, hotkey: bt.Keypair
) -> Optional[requests.Response]:
    """
    Log competition metrics to the centralized logging server.
    """
    try:
        metrics["validator_key"] = hotkey.ss58_address

        # some metrics are nested under the hotkey, so we need to move them to the competitors list
        # for example `hotkey.0x374672364.historical.improvement_rate` isn't really good for BigQuery
        # so preparing miners data to go to a separate table
        metrics["competitors"] = {}
        for key in metrics.keys():
            if not key.startswith("hotkey."):
                continue
            hotkey = key.split(".")[1]
            if metrics["competitors"].get(hotkey) is None:
                metrics["competitors"][hotkey] = {
                    "hotkey": hotkey,
                }
            # something like `hotkey.{miner_hotkey}.historical.improvement_rate` turns into
            # `historical_improvement_rate` in the competitors dict
            metrics["competitors"][hotkey]["_".join(key.split(".")[2:])] = metrics.pop(key)
        metrics["competitors"] = list(metrics["competitors"].values())

        input_bytes = json.dumps(metrics).encode("utf-8")
        # sign the inputs with your hotkey
        signature = hotkey.sign(input_bytes)
        # encode the inputs and signature as base64
        signature_str = base64.b64encode(signature).decode("utf-8")

        return session.post(
            COMPETITION_LOGGING_URL,
            data=input_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Request-Signature": signature_str,
            },
            timeout=5,
        )
    except requests.exceptions.RequestException as e:
        bt.logging.error(f"Failed to log competition metrics: {e}")
        return None
