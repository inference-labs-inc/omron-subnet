#!/usr/bin/env python3

"""
This script is used to convert the Reward function into a zk circuit.
Validators produce reward proofs that can be verified by anyone.
"""

import argparse
import json
import os
import sys

# Importing necessary libraries
import bittensor as bt
import ezkl
import numpy as np
import onnxruntime as ort
import torch

# Parse command line arguments
parser = argparse.ArgumentParser(description="Circuitize Reward Function")
parser.add_argument("--trace", action="store_true", help="Enable trace logging")
args = parser.parse_args()

# Enable trace logging if --trace is specified
if args.trace:
    bt.logging.set_trace(True)

# Adding the parent directory to the system path for module import
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _validator.reward import Reward

# Changing the working directory to the reward folder in the deployment layer
os.chdir(os.path.join(os.path.dirname(__file__), "..", "deployment_layer", "reward"))

# Instantiate the Reward model and set it to evaluation mode
reward_model = Reward()
reward_model.eval()
bt.logging.trace(
    f"Instantiated Reward model: {reward_model} and set it to evaluation mode."
)

# Define dummy inputs for the model export
dummy_inputs = {
    "max_score": torch.tensor([1.0], dtype=torch.float32),
    "score": torch.tensor([0.5], dtype=torch.float32),
    "verification_result": torch.tensor([1.0], dtype=torch.float32),
    "factor": torch.tensor([1.0], dtype=torch.float32),
}
bt.logging.trace(f"Defined dummy inputs for the model export: {dummy_inputs}.")

if __name__ == "__main__":
    bt.logging.info("Creating zk circuit for the reward function")

    # Exporting the model to ONNX format
    bt.logging.trace("Starting the export of the model to ONNX format.")
    torch.onnx.export(
        reward_model,
        tuple(dummy_inputs.values()),
        "network.onnx",
        opset_version=11,
        input_names=list(dummy_inputs.keys()),
        output_names=["output"],
    )
    bt.logging.trace("Completed exporting the model to ONNX format: network.onnx.")

    # Running inference with ONNX Runtime
    bt.logging.trace("Initiating inference with ONNX Runtime on network.onnx.")
    ort_session = ort.InferenceSession("network.onnx")
    np_inputs = {k: np.array(v, dtype=np.float32) for k, v in dummy_inputs.items()}
    output = ort_session.run(
        output_names=["output"],
        input_feed=np_inputs,
    )
    bt.logging.trace(f"Completed inference with ONNX Runtime. Output: {output}.")

    # Saving input and output data to JSON file
    bt.logging.trace("Saving the input and output data to a JSON file: input.json.")
    with open("input.json", "w", encoding="utf8") as input_file:
        input_json_object = {
            "input_data": [v.tolist() for v in dummy_inputs.values()],
            "output_data": [output[0].tolist()],
        }
        json_str = json.dumps(input_json_object, indent=4)
        input_file.write(json_str)
    bt.logging.trace(
        "Saved the input and output data to a JSON file successfully: input.json."
    )

    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "public"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "fixed"
    ezkl.gen_settings(py_run_args=py_run_args)
    ezkl.calibrate_settings("input.json", target="accuracy")
    ezkl.compile_circuit()
    ezkl.get_srs()
    ezkl.setup(
        model="model.compiled", vk_path="vk.key", pk_path="pk.key", srs_path="kzg.srs"
    )
    ezkl.gen_witness(
        data="input.json",
        model="model.compiled",
        output="witness.json",
        vk_path="vk.key",
        srs_path="kzg.srs",
    )
    ezkl.prove(witness="witness.json", model="model.compiled")
    ezkl.verify()

    bt.logging.success("Successfully circuitized the reward function.")
