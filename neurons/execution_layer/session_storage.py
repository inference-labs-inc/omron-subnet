import os
import bittensor as bt
from dataclasses import dataclass, field
from utils.system import get_temp_folder
from execution_layer.circuit_metadata import ProofSystem

dir_path = os.path.dirname(os.path.realpath(__file__))


@dataclass
class SessionStorage:
    model_id: str
    session_uuid: str
    base_path: str = field(default_factory=get_temp_folder)
    input_path: str = field(init=False)
    witness_path: str = field(init=False)
    proof_path: str = field(init=False)

    def __post_init__(self):
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        self.input_path = os.path.join(
            self.base_path, f"input_{self.model_id}_{self.session_uuid}.json"
        )
        self.witness_path = os.path.join(
            self.base_path, f"witness_{self.model_id}_{self.session_uuid}.json"
        )
        self.proof_path = os.path.join(
            self.base_path, f"proof_{self.model_id}_{self.session_uuid}.json"
        )
        bt.logging.debug(
            f"SessionStorage initialized with model_id: {self.model_id} and session_uuid: {self.session_uuid}"
        )
        bt.logging.trace(f"Input path: {self.input_path}")
        bt.logging.trace(f"Witness path: {self.witness_path}")
        bt.logging.trace(f"Proof path: {self.proof_path}")

    def get_proof_path_for_iteration(self, iteration: int) -> str:
        return os.path.join(
            self.base_path,
            f"proof_{self.model_id}_{self.session_uuid}_{iteration}.json",
        )

    def get_session_path(self, session_id: str) -> str:
        session_path = os.path.join(self.base_path, session_id)
        if not os.path.exists(session_path):
            os.makedirs(session_path)
        return session_path


class EZKLSessionStorage(SessionStorage):
    def __post_init__(self):
        super().__post_init__()
        bt.logging.trace("EZKLSessionStorage: Using standard EZKL file paths")


class CircomSessionStorage(SessionStorage):
    public_path: str = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.public_path = os.path.join(
            self.base_path, f"proof_{self.model_id}_{self.session_uuid}.public.json"
        )
        bt.logging.trace(f"CircomSessionStorage: Public path: {self.public_path}")


class DCAPSessionStorage(SessionStorage):
    output_path: str = field(init=False)
    quote_path: str = field(init=False)

    def __post_init__(self):
        # Initialize base path directory (from parent's __post_init__)
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        # Initialize paths required by base class, excluding witness_path (handled by property)
        self.input_path = os.path.join(
            self.base_path, f"input_{self.model_id}_{self.session_uuid}.json"
        )
        self.proof_path = os.path.join(
            self.base_path, f"proof_{self.model_id}_{self.session_uuid}.json"
        )
        # Initialize DCAP-specific paths
        self.output_path = os.path.join(
            self.base_path, f"output_{self.model_id}_{self.session_uuid}.json"
        )
        self.quote_path = os.path.join(
            self.base_path, f"quote_{self.model_id}_{self.session_uuid}.bin"
        )
        bt.logging.debug(
            f"DCAPSessionStorage initialized with model_id: {self.model_id} and session_uuid: {self.session_uuid}"
        )
        bt.logging.trace(f"DCAPSessionStorage: Input path: {self.input_path}")
        bt.logging.trace(f"DCAPSessionStorage: Output path: {self.output_path}")
        bt.logging.trace(f"DCAPSessionStorage: Quote path: {self.quote_path}")
        bt.logging.trace(f"DCAPSessionStorage: Proof path: {self.proof_path}")
        bt.logging.trace(f"DCAPSessionStorage: Witness path: {self.witness_path}")

    @property
    def witness_path(self) -> str:
        return self.output_path


class SessionStorageFactory:
    _storage_classes = {
        ProofSystem.EZKL: EZKLSessionStorage,
        ProofSystem.CIRCOM: CircomSessionStorage,
        ProofSystem.DCAP: DCAPSessionStorage,
    }

    @classmethod
    def create_storage(
        cls, proof_system: ProofSystem, model_id: str, session_uuid: str
    ) -> SessionStorage:
        if isinstance(proof_system, str):
            try:
                proof_system = ProofSystem[proof_system.upper()]
            except KeyError as e:
                raise ValueError(f"Invalid proof system string: {proof_system}") from e

        storage_class = cls._storage_classes.get(proof_system)
        if storage_class is None:
            bt.logging.warning(
                f"No specific storage for {proof_system}, using base SessionStorage"
            )
            storage_class = SessionStorage

        return storage_class(model_id, session_uuid)
