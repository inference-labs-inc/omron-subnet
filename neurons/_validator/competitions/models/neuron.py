from pydantic import BaseModel


class NeuronState(BaseModel):
    hotkey: str
    uid: int
    score: float
    proof_size: int
    response_time: float
    verification_result: bool
    accuracy: float
    hash: str
