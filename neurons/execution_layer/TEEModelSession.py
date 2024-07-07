"""
Session for running models within a TEE
Provides the basis for processing an inference of LLaMa-7B within an Intel SGX TEE
"""

import time
from typing import Any
from attr import field
import requests
from dataclasses import dataclass

# URL for the local FastChat (BigDL SGX)
OPENAPI_URL = "http://localhost:8000"


@dataclass
class InferenceResult:
    result: Any
    success: bool
    execution_time: float


class TEESession:
    """
    Trusted Execution Session
    """

    endpoint: str = field(init=False)

    def __init__(self):
        self.endpoint = f"{OPENAPI_URL}/v1/completions"

    def run_verified_inference(self, inputs):
        """
        Run a verified inference via TEE
        Determine inference timing, and the result. A failure can indicate:
        - A timeout
        - A failed attestation / verification
        - Another error preventing the miner from responding correctly
        """
        start_time = time.time()

        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": "llama",
                    "prompt": inputs,
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            success = True
        except requests.RequestException as e:
            result = str(e)
            success = False

        end_time = time.time()
        execution_time = end_time - start_time

        return InferenceResult(
            result=result, success=success, execution_time=execution_time
        )
