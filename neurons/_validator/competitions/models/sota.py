from dataclasses import dataclass


@dataclass
class SotaState:
    score: float = 0.0
    hash: str | None = None
    hotkey: str | None = None
    proof_size: float = float("inf")
    response_time: float = float("inf")
    timestamp: int = 0
    accuracy: float = 0.0
