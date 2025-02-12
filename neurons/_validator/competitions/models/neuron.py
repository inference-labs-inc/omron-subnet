from dataclasses import dataclass


@dataclass
class NeuronState:
    hotkey: str
    uid: int
    sota_relative_score: float
    proof_size: float
    response_time: float
    verification_result: bool
    raw_accuracy: float
    hash: str
