#!/usr/bin/env python3

import json
import os
import sys

import ezkl
import numpy as np
import onnxruntime as ort
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _validator.reward import Reward

os.chdir(os.path.join(os.path.dirname(__file__), "..", "deployment_layer", "reward"))
import torch.onnx

# Instantiate the model
reward_model = Reward()

# Set the model to evaluation mode
reward_model.eval()

# Define dummy inputs for the export
dummy_max_score = torch.tensor([1.0], dtype=torch.float32)
dummy_score = torch.tensor([0.5], dtype=torch.float32)
dummy_verification_result = torch.tensor([1.0], dtype=torch.float32)
dummy_factor = torch.tensor([1.0], dtype=torch.float32)

if __name__ == "__main__":
    torch.onnx.export(
        reward_model,
        (
            dummy_max_score,
            dummy_score,
            dummy_verification_result,
            dummy_factor,
        ),
        "network.onnx",
        opset_version=11,
        input_names=[
            "max_score",
            "score",
            "verification_result",
            "factor",
        ],
        output_names=["new_score"],
    )
    ort_session = ort.InferenceSession("network.onnx")
    output = ort_session.run(
        output_names=["new_score"],
        input_feed={
            "max_score": np.array([1.0], dtype=np.float32),
            "score": np.array([0.5], dtype=np.float32),
            "verification_result": np.array([1.0], dtype=np.float32),
            "factor": np.array([1.0], dtype=np.float32),
        },
    )
    print(output)
    with open("input.json", "w", encoding="utf8") as input_file:
        input_json_object = {
            "input_data": [
                dummy_max_score.tolist(),
                dummy_score.tolist(),
                dummy_verification_result.tolist(),
                dummy_factor.tolist(),
            ],
            "output_data": [output[0].tolist()],
        }
        json_str = json.dumps(input_json_object)
        input_file.write(json_str)
        py_run_args = ezkl.PyRunArgs()
        py_run_args.input_visibility = "public"
        py_run_args.output_visibility = "public"
        py_run_args.param_visibility = "fixed"
    ezkl.gen_settings(py_run_args=py_run_args)
    ezkl.calibrate_settings("input.json")
    ezkl.compile_circuit()
    ezkl.get_srs()

    ezkl.setup()
    ezkl.gen_witness()
    ezkl.prove()
    ezkl.verify()
