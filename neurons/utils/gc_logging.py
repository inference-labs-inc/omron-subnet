import base64
import json
import os
import time
from typing import Any, Dict, List

import bittensor as bt
import psutil
import requests
import torch

from _validator.models.miner_response import MinerResponse

LOGGING_URL = os.getenv(
    "LOGGING_URL",
    "https://api.omron.ai/statistics/log/",
)


def get_system_metrics() -> Dict[str, Any]:
    """Get system performance metrics."""
    process = psutil.Process()
    return {
        "cpu_percent": process.cpu_percent(),
        "memory_percent": process.memory_percent(),
        "memory_info": dict(process.memory_info()._asdict()),
        "num_threads": process.num_threads(),
        "disk_usage": psutil.disk_usage("/").percent,
        "network_io": dict(psutil.net_io_counters()._asdict()),
    }


def prepare_response_metrics(
    responses: List[MinerResponse],
    metagraph: bt.metagraph,
) -> List[Dict[str, Any]]:
    """Prepare detailed response metrics for logging."""
    return [
        {
            "miner_key": response.axon.hotkey,
            "miner_uid": response.uid,
            "stake": float(metagraph.S[response.uid].item()),
            "trust": float(metagraph.T[response.uid].item()),
            "consensus": float(metagraph.C[response.uid].item()),
            "incentive": float(metagraph.I[response.uid].item()),
            "dividends": float(metagraph.D[response.uid].item()),
            "emission": float(metagraph.E[response.uid].item()),
            "response_time": response.response_time,
            "verification_time": response.verification_time,
            "proof_size": len(response.proof_content) if response.proof_content else 0,
            "verification_success": response.verification_result,
            "error": response.error if response.error else None,
            "model_name": (
                response.circuit.metadata.name if response.circuit else "unknown"
            ),
            "timestamp": int(time.time()),
            **response.to_log_dict(metagraph),
        }
        for response in responses
    ]


def log_responses(
    metagraph: bt.metagraph,
    hotkey: bt.Keypair,
    uid: int,
    responses: List[MinerResponse],
    overhead_time: float,
    block: int,
    scores: torch.Tensor,
) -> None:
    """
    Log comprehensive validator metrics to the centralized logging server.

    Args:
        metagraph: Network metagraph containing stake and trust info
        hotkey: Validator hotkey for signing
        uid: Validator UID
        responses: List of miner responses
        overhead_time: Time spent on overhead operations
        block: Current block number
        scores: Current scores tensor
    """
    try:
        response_metrics = prepare_response_metrics(responses, metagraph)

        data = {
            "validator_key": hotkey.ss58_address,
            "validator_uid": uid,
            "block": block,
            "timestamp": int(time.time()),
            # Performance metrics
            "overhead_duration": overhead_time,
            "total_responses": len(responses),
            "successful_verifications": sum(
                1 for r in responses if r.verification_result
            ),
            "avg_response_time": (
                sum(r.response_time for r in responses) / len(responses)
                if responses
                else 0
            ),
            "avg_verification_time": (
                sum(r.verification_time for r in responses) / len(responses)
                if responses
                else 0
            ),
            "avg_proof_size": (
                sum(len(r.proof_content) if r.proof_content else 0 for r in responses)
                / len(responses)
                if responses
                else 0
            ),
            # System metrics
            "system_metrics": get_system_metrics(),
            # Detailed response data
            "responses": response_metrics,
            # Network metrics
            "scores": {
                int(uid): float(value.item())
                for uid, value in enumerate(scores)
                if not torch.isnan(value)
            },
            # Error tracking
            "errors": [
                {
                    "miner_uid": r.uid,
                    "error_type": r.error,
                    "timestamp": int(time.time()),
                }
                for r in responses
                if r.error
            ],
        }

        input_bytes = json.dumps(data).encode("utf-8")
        signature = hotkey.sign(input_bytes)
        signature_str = base64.b64encode(signature).decode("utf-8")

        resp = requests.post(
            LOGGING_URL,
            data=input_bytes,
            headers={
                "X-Request-Signature": signature_str,
                "Content-Type": "application/json",
            },
            timeout=10,  # Add timeout to prevent hanging
        )
        resp.raise_for_status()

    except requests.exceptions.RequestException as e:
        bt.logging.error(f"Failed to log responses: {str(e)}")
    except Exception as e:
        bt.logging.error(f"Unexpected error logging responses: {str(e)}")
