from __future__ import annotations
import os
import traceback
import bittensor as bt
from execution_layer.circuit import Circuit


class CircuitStore:
    """
    A class to manage and store Circuit objects.

    This class is responsible for loading, storing, and retrieving Circuit objects.
    """

    def __init__(self):
        """
        Initialize the CircuitStore.

        Creates an empty dictionary to store Circuit objects and loads circuits.
        """
        self.circuits: dict[str, Circuit] = {}
        self.load_circuits()

    def load_circuits(self):
        """
        Load circuits from the file system.

        Searches for directories starting with 'model_' in the deployment layer path,
        attempts to create Circuit objects from these directories, and stores them
        in the circuits dictionary.
        """
        deployment_layer_path = os.path.dirname(__file__)
        bt.logging.info(f"Loading circuits from {deployment_layer_path}")

        for folder_name in os.listdir(deployment_layer_path):
            folder_path = os.path.join(deployment_layer_path, folder_name)

            if os.path.isdir(folder_path) and folder_name.startswith("model_"):
                circuit_id = folder_name.split("_")[1]
                try:
                    bt.logging.debug(f"Attempting to load circuit {circuit_id}")
                    circuit = Circuit(circuit_id)
                    self.circuits[circuit_id] = circuit
                    bt.logging.info(f"Successfully loaded circuit {circuit_id}")
                except Exception as e:
                    bt.logging.error(f"Error loading circuit {circuit_id}: {e}")
                    traceback.print_exc()
                    continue

        bt.logging.info(f"Loaded {len(self.circuits)} circuits")

    def get_circuit(self, circuit_id: str) -> Circuit | None:
        """
        Retrieve a Circuit object by its ID.

        Args:
            circuit_id (str): The ID of the circuit to retrieve.

        Returns:
            Circuit | None: The Circuit object if found, None otherwise.
        """
        circuit = self.circuits.get(circuit_id)
        if circuit:
            bt.logging.debug(f"Retrieved circuit {circuit_id}")
        else:
            bt.logging.warning(f"Circuit {circuit_id} not found")
        return circuit

    def list_circuits(self) -> list[str]:
        """
        Get a list of all circuit IDs.

        Returns:
            list[str]: A list of circuit IDs.
        """
        circuit_list = list(self.circuits.keys())
        bt.logging.debug(f"Listed {len(circuit_list)} circuits")
        return circuit_list


circuit_store = CircuitStore()
bt.logging.info("CircuitStore initialized")
