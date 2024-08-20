import os
import bittensor as bt

from attr import define, field

dir_path = os.path.dirname(os.path.realpath(__file__))


@define
class SessionStorage:
    model_id: str = field()
    session_uuid: str = field()
    base_path: str = field(default=os.path.join(dir_path, "temp"))
    input_path: str = field(init=False)
    witness_path: str = field(init=False)
    proof_path: str = field(init=False)
    aggregated_proof_path: str = field(init=False)
    public_path: str = field(init=False)

    def __attrs_post_init__(self):
        self.input_path = os.path.join(
            self.base_path, f"input_{self.model_id}_{self.session_uuid}.json"
        )
        self.witness_path = os.path.join(
            self.base_path, f"witness_{self.model_id}_{self.session_uuid}.json"
        )
        self.proof_path = os.path.join(
            self.base_path, f"proof_{self.model_id}_{self.session_uuid}.json"
        )
        self.aggregated_proof_path = os.path.join(
            self.base_path, f"aggregated_proof_{self.model_id}_{self.session_uuid}.json"
        )
        self.public_path = os.path.join(
            self.base_path, f"proof_{self.model_id}_{self.session_uuid}.public.json"
        )
        bt.logging.debug(
            f"SessionStorage initialized with model_id: {self.model_id} and session_uuid: {self.session_uuid}"
        )
        bt.logging.trace(f"Input path: {self.input_path}")
        bt.logging.trace(f"Witness path: {self.witness_path}")
        bt.logging.trace(f"Proof path: {self.proof_path}")
        bt.logging.trace(f"Aggregated proof path: {self.aggregated_proof_path}")

    def get_proof_path_for_iteration(self, iteration: int) -> str:
        return os.path.join(
            self.base_path,
            f"proof_{self.model_id}_{self.session_uuid}_{iteration}.json",
        )
