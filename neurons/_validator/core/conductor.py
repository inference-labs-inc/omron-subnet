from neurons.execution_layer.base_input import BaseInput
from neurons.execution_layer.circuit import Circuit
from neurons.execution_layer.verified_model_session import VerifiedModelSession
from neurons.deployment_layer.circuit_store import circuit_store


class Conductor:

    circuit: Circuit
    inputs: BaseInput

    def __init__(self, circuit: Circuit, inputs: BaseInput):
        self.circuit = circuit
        self.inputs = inputs

    def orchestrate(self):
        layer_map = self.circuit.get_layer_map()
        witness_map = {}

        for layer_id, layer in layer_map.items():
            session = VerifiedModelSession(
                self.inputs, circuit_store.get_circuit(layer.circuit_id)
            )
            witness = session.generate_witness(True)
            witness_map[layer_id] = witness
