from __future__ import annotations
from pydantic import BaseModel
from execution_layer.base_input import BaseInput
from execution_layer.input_registry import InputRegistry
from _validator.utils.request_type import RequestType
import random
import secrets

BATCH_SIZE = 256
POW_MIN_DIFFICULTY = 1
POW_MAX_DIFFICULTY = 8
POW_TIMEOUT = 10.0
SUCCESS_WEIGHT = 0.3
DIFFICULTY_WEIGHT = 0.2
TIME_ELAPSED_WEIGHT = 0.2
FAILED_PENALTY_WEIGHT = 0.2
ALLOCATION_WEIGHT = 0.1
MAX_SCORE_CHALLENGE = 1.0
MAX_SCORE = 1.0
FAILED_PENALTY_EXP = 1.0
HALF_VALIDATORS = 4.5


class CircuitInputSchema(BaseModel):
    success_weight: list[float]
    difficulty_weight: list[float]
    time_elapsed_weight: list[float]
    failed_penalty_weight: list[float]
    allocation_weight: list[float]
    pow_min_difficulty: list[int]
    pow_max_difficulty: list[int]
    pow_timeout: list[float]
    max_score_challenge: list[float]
    max_score: list[float]
    failed_penalty_exp: list[float]
    challenge_attempts: list[int]
    challenge_successes: list[int]
    last_20_challenge_failed: list[int]
    challenge_elapsed_time_avg: list[float]
    challenge_difficulty_avg: list[float]
    has_docker: list[bool]
    allocated_hotkey: list[bool]
    penalized_hotkey_count: list[int]
    half_validators: list[float]
    nonce: list[int]


@InputRegistry.register(
    "1d60d545b7c5123fd60524dcbaf57081ca7dc4a9ec36c892927a3153328d17c0"
)
class CircuitInput(BaseInput):
    def __init__(
        self, request_type: RequestType, data: dict[str, object] | None = None
    ):
        super().__init__(request_type, data)

    @staticmethod
    def generate() -> dict[str, object]:
        inputs = []
        for _ in range(BATCH_SIZE):
            attempts = random.randint(1, 7)
            inputs.append(
                {
                    "challenge_attempts": attempts,
                    "challenge_successes": random.randint(0, attempts),
                    "last_20_challenge_failed": random.randint(0, 20),
                    "challenge_elapsed_time_avg": random.uniform(0.001, POW_TIMEOUT),
                    "challenge_difficulty_avg": random.uniform(
                        POW_MIN_DIFFICULTY, POW_MAX_DIFFICULTY
                    ),
                    "has_docker": random.choice([True, False]),
                    "allocated_hotkey": random.choice([True, False]),
                    "penalized_hotkey_count": random.randint(0, 3),
                }
            )

        return {
            "success_weight": [SUCCESS_WEIGHT],
            "difficulty_weight": [DIFFICULTY_WEIGHT],
            "time_elapsed_weight": [TIME_ELAPSED_WEIGHT],
            "failed_penalty_weight": [FAILED_PENALTY_WEIGHT],
            "allocation_weight": [ALLOCATION_WEIGHT],
            "pow_min_difficulty": [POW_MIN_DIFFICULTY],
            "pow_max_difficulty": [POW_MAX_DIFFICULTY],
            "pow_timeout": [POW_TIMEOUT],
            "max_score_challenge": [MAX_SCORE_CHALLENGE],
            "max_score": [MAX_SCORE],
            "failed_penalty_exp": [FAILED_PENALTY_EXP],
            "challenge_attempts": [x["challenge_attempts"] for x in inputs],
            "challenge_successes": [x["challenge_successes"] for x in inputs],
            "last_20_challenge_failed": [x["last_20_challenge_failed"] for x in inputs],
            "challenge_elapsed_time_avg": [
                x["challenge_elapsed_time_avg"] for x in inputs
            ],
            "challenge_difficulty_avg": [x["challenge_difficulty_avg"] for x in inputs],
            "has_docker": [x["has_docker"] for x in inputs],
            "allocated_hotkey": [x["allocated_hotkey"] for x in inputs],
            "penalized_hotkey_count": [x["penalized_hotkey_count"] for x in inputs],
            "half_validators": [4.5],
            "nonce": [secrets.randbits(32)],
        }

    @staticmethod
    def validate(data: dict[str, object]) -> None:
        return CircuitInputSchema(**data)

    @staticmethod
    def process(data: dict[str, object]) -> dict[str, object]:
        """
        Add a random nonce to ensure that the request is not reused.
        """
        data["nonce"] = secrets.randbits(32)
        return data
