from __future__ import annotations

import os
import traceback
from typing import Optional

import bittensor as bt
from packaging import version

import cli_parser
from constants import IGNORED_MODEL_HASHES
from execution_layer.circuit import Circuit


class CircuitStore:
    """
    A Singleton class to manage and store Circuit objects.

    This class is responsible for loading, storing, and retrieving Circuit objects.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Override the __new__ method to implement the Singleton pattern.
        """
        if not cls._instance:
            cls._instance = super(CircuitStore, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        """
        Initialize the CircuitStore.

        Creates an empty dictionary to store Circuit objects and loads circuits.
        """
        self.circuits: dict[str, Circuit] = {}

    def load_circuits(self, deployment_layer_path: Optional[str] = None):
        """
        Load circuits from the file system.

        Searches for directories starting with 'model_' in the deployment layer path,
        attempts to create Circuit objects from these directories, and stores them
        in the circuits dictionary.
        """
        deployment_layer_path = (
            deployment_layer_path or cli_parser.config.external_model_dir
        )
        bt.logging.info(f"Loading circuits from {deployment_layer_path}")

        for folder_name in os.listdir(deployment_layer_path):
            folder_path = os.path.join(deployment_layer_path, folder_name)

            if os.path.isdir(folder_path) and folder_name.startswith("model_"):
                circuit_id = folder_name.split("_")[1]

                if circuit_id in IGNORED_MODEL_HASHES:
                    bt.logging.info(f"Ignoring circuit {circuit_id}")
                    continue

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
            bt.logging.debug(f"Retrieved circuit {circuit}")
        else:
            bt.logging.warning(f"Circuit {circuit_id} not found")
        return circuit

    def get_latest_circuit_for_netuid(self, netuid: int):
        """
        Get the latest circuit for a given netuid by comparing semver version strings.

        Args:
            netuid (int): The subnet ID to find the latest circuit for

        Returns:
            Circuit | None: The circuit with the highest semver version for the given netuid,
            or None if no circuits found
        """

        matching_circuits = [
            c for c in self.circuits.values() if c.metadata.netuid == netuid
        ]
        if not matching_circuits:
            return None

        return max(matching_circuits, key=lambda c: version.parse(c.metadata.version))

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
