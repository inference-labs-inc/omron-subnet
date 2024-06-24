#!/usr/bin/env python3

"""
This script is used to convert the Reward function into a zk circuit.
Validators produce reward proofs that can be verified by anyone.
"""


import argparse
import asyncio
import hashlib
import json
import os
import sys
from importlib.metadata import version

# Importing necessary libraries
import bittensor as bt
import ezkl
import numpy as np
import onnxruntime as ort
import torch

print(f"Using EZKL Version: {version('ezkl')}")

PRODUCE_AGGREGATE_CIRCUIT = False

# Append to system path in order to import validator
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# trunk-ignore(flake8/E402)
# trunk-ignore(pylint/C0413)
from _validator.reward import Reward

# Parse command line arguments
parser = argparse.ArgumentParser(description="Circuitize Reward Function")
parser.add_argument("--trace", action="store_true", help="Enable trace logging")
args = parser.parse_args()

# Enable trace logging if --trace is specified
if args.trace:
    bt.logging.set_trace(True)

# Create the new circuit's directory
os.chdir(os.path.join(os.path.dirname(__file__), "..", "deployment_layer"))
new_model_folder = "new_model"
os.makedirs(new_model_folder, exist_ok=True)
os.chdir(new_model_folder)

# Instantiate the Reward model and set it to evaluation mode
reward_model = Reward()
reward_model.eval()
bt.logging.trace(
    f"Instantiated Reward model: {reward_model} and set it to evaluation mode."
)

SETTINGS_CONFIGURATION = {
    "run_args": {
        "tolerance": {"val": 0.0, "scale": 1.0},
        "input_scale": 16,
        "param_scale": 16,
        "scale_rebase_multiplier": 1,
        "lookup_range": [-150354, 97480],
        "logrows": 18,
        "num_inner_cols": 2,
        "variables": [["batch_size", 1]],
        "input_visibility": "Public",
        "output_visibility": "Public",
        "param_visibility": "Fixed",
        "div_rebasing": False,
        "rebase_frac_zero_constants": False,
        "check_mode": "UNSAFE",
        "commitment": "KZG",
    },
    "num_rows": 112,
    "total_assignments": 224,
    "total_const_size": 11,
    "total_dynamic_col_size": 0,
    "num_dynamic_lookups": 0,
    "num_shuffles": 0,
    "total_shuffle_col_size": 0,
    "model_instance_shapes": [
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [64],
        [1],
        [1],
        [1],
        [64],
        [1],
        [1],
    ],
    "model_output_scales": [16, 0, 0, 0],
    "model_input_scales": [16, 16, 0, 0, 16, 16, 16, 0, 0, 0],
    "module_sizes": {"polycommit": [], "poseidon": [0, [0]]},
    "required_lookups": [
        "ReLU",
        {"Min": {"scale": 65536.0, "a": 0.99}},
        {"Tan": {"scale": 65536.0}},
        {"Min": {"scale": 65536.0, "a": 1.0}},
    ],
    "required_range_checks": [[-32768, 32768], [32768, 98304]],
    "check_mode": "UNSAFE",
    "version": "10.2.7",
    "num_blinding_factors": None,
    "timestamp": 1718658006597,
}


# Define sample inputs for the model export
sample_inputs = {
    "maximum_score": torch.tensor([0.008], dtype=torch.float32),
    "previous_score": torch.tensor([0], dtype=torch.float32),
    "verified": torch.tensor([True], dtype=torch.bool),
    "proof_size": torch.tensor([3648], dtype=torch.int),
    "response_time": torch.tensor([3], dtype=torch.float32),
    "maximum_response_time": torch.tensor([20], dtype=torch.float32),
    "minimum_response_time": torch.tensor([2], dtype=torch.float32),
    "validator_hotkey": torch.tensor(
        [
            57,
            54,
            100,
            54,
            101,
            55,
            57,
            97,
            56,
            98,
            50,
            48,
            49,
            57,
            48,
            99,
            52,
            54,
            50,
            98,
            100,
            99,
            53,
            97,
            49,
            97,
            53,
            102,
            101,
            51,
            57,
            99,
            99,
            51,
            55,
            53,
            52,
            55,
            52,
            55,
            52,
            50,
            53,
            50,
            56,
            100,
            53,
            101,
            54,
            54,
            50,
            57,
            51,
            55,
            57,
            50,
            99,
            100,
            100,
            97,
            101,
            100,
            50,
            99,
        ],
        dtype=torch.int,
    ),
    "block_number": torch.tensor([2080000], dtype=torch.int),
    "miner_uid": torch.tensor([110], dtype=torch.int),
}
bt.logging.trace(f"Defined sample inputs for the model export: {sample_inputs}.")

if __name__ == "__main__":
    bt.logging.info("Creating zk circuit for the reward function")

    # Export the model with sample inputs to ONNX format
    bt.logging.trace("Starting the export of the model to ONNX format.")
    torch.onnx.export(
        reward_model,
        tuple(sample_inputs.values()),
        "network.onnx",
        opset_version=11,
        input_names=list(sample_inputs.keys()),
        output_names=["output"],
    )
    bt.logging.trace("Completed exporting the model to ONNX format: network.onnx.")

    # Running inference with ONNX Runtime
    bt.logging.trace("Initiating inference with ONNX Runtime on network.onnx.")
    ort_session = ort.InferenceSession("network.onnx")
    np_inputs = {
        k: (
            np.array(
                v,
                dtype=(
                    np.float32
                    if k
                    not in [
                        "block_number",
                        "miner_uid",
                        "proof_size",
                        "validator_hotkey",
                    ]
                    else np.int32
                ),
            )
            if k != "verified"
            else np.array(v, dtype=np.bool_)
        )
        for k, v in sample_inputs.items()
    }
    output = ort_session.run(
        output_names=["output"],
        input_feed=np_inputs,
    )
    bt.logging.trace(f"Completed inference with ORT. Output: {output}.")

    # Save input, input shapes and output into an input.json
    bt.logging.trace("Saving the input and output data to a JSON file: input.json.")
    with open("input.json", "w", encoding="utf8") as input_file:
        input_json_object = {
            "input_data": [v.tolist() for v in sample_inputs.values()],
            "input_shapes": [v.shape for v in sample_inputs.values()],
            "output_data": [output[0].tolist()],
        }
        json_str = json.dumps(input_json_object, indent=4)
        input_file.write(json_str)
    bt.logging.trace(
        "Saved the input and output data to a JSON file successfully: input.json."
    )

    async def circuitize():
        py_run_args = ezkl.PyRunArgs()
        py_run_args.input_visibility = "public"
        py_run_args.output_visibility = "public"
        py_run_args.param_visibility = "fixed"
        # ezkl.gen_settings(py_run_args=py_run_args)
        # ezkl.calibrate_settings("input.json", target="accuracy", scales=[16])
        # Use predefined settings rather than generated settings
        with open("settings.json", "w") as f:
            json.dump(SETTINGS_CONFIGURATION, f)
        ezkl.get_srs()
        ezkl.compile_circuit()
        ezkl.setup(
            model="model.compiled",
            vk_path="vk.key",
            pk_path="pk.key",
        )
        ezkl.gen_witness(
            data="input.json",
            model="model.compiled",
            output="witness.json",
            vk_path="vk.key",
        )
        ezkl.prove(
            witness="witness.json",
            model="model.compiled",
            pk_path="pk.key",
            proof_path="proof.json",
            proof_type="for-aggr" if PRODUCE_AGGREGATE_CIRCUIT else "single",
        )
        ezkl.verify(
            proof_path="proof.json",
            settings_path="settings.json",
            vk_path="vk.key",
        )
        ezkl.prove(
            witness="witness.json",
            model="model.compiled",
            pk_path="pk.key",
            proof_path="proof_2.json",
            proof_type="for-aggr" if PRODUCE_AGGREGATE_CIRCUIT else "single",
        )
        if PRODUCE_AGGREGATE_CIRCUIT:
            ezkl.setup_aggregate(
                sample_snarks=["proof.json", "proof_2.json"], logrows=23
            )
            ezkl.aggregate(
                aggregation_snarks=["proof.json", "proof_2.json"],
                proof_path="proof_aggr.json",
                vk_path="pk_aggr.key",
                logrows=22,
            )
            ezkl.verify_aggr(
                proof_path="proof_aggr.json",
                logrows=22,
                vk_path="vk_aggr.key",
            )

        # Hash the VK and set it as the folder's name
        with open("vk.key", "rb") as f:
            vk_key = f.read()
        model_vk_key_sha256 = hashlib.sha256(vk_key).hexdigest()
        new_folder_name = f"model_{model_vk_key_sha256}"
        parent_dir = os.path.dirname(os.getcwd())
        new_path = os.path.join(parent_dir, new_folder_name)
        os.rename(os.getcwd(), new_path)
        bt.logging.success("Successfully circuitized the reward function.")


loop = asyncio.get_event_loop()
loop.run_until_complete(circuitize())
