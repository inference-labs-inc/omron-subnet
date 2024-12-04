import hashlib
from execution_layer.generic_input import GenericInput


def hash_inputs(inputs: GenericInput) -> str:
    """
    Hashes inputs to proof of weights, excluding dynamic fields.

    Args:
        inputs (dict): The inputs to hash.

    Returns:
        str: The hashed inputs.
    """
    filtered_inputs = {
        k: v
        for k, v in inputs.to_dict().items()
        if k not in ["validator_uid", "nonce", "uid_responsible_for_proof"]
    }
    return hashlib.sha256(str(filtered_inputs).encode()).hexdigest()
