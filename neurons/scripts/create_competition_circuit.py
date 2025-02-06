import torch
import torch.nn as nn
import torch.onnx

import subprocess
import logging


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        x = self.conv(x)
        stage1_delta_k = torch.ones(x.shape[0], 1)
        prob_stage_1 = torch.ones(x.shape[0], 3)
        index_offset_stage1 = torch.ones(x.shape[0], 3)
        return stage1_delta_k, prob_stage_1, index_offset_stage1


model = DummyModel()
dummy_input = torch.randn(1, 3, 64, 64)

torch.onnx.export(
    model,
    dummy_input,
    "network.onnx",
    input_names=["input"],
    output_names=["stage1_delta_k", "prob_stage_1", "index_offset_stage1"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "stage1_delta_k": {0: "batch_size"},
        "prob_stage_1": {0: "batch_size"},
        "index_offset_stage1": {0: "batch_size"},
    },
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

commands = [
    "ezkl gen-settings",
    "ezkl calibrate-settings --data ./neurons/_validator/competitions/1/input.json",
    "ezkl compile-circuit",
    "ezkl setup",
    "ezkl gen-witness --data ./neurons/_validator/competitions/1/input.json",
    "ezkl prove",
    "ezkl verify",
]

for cmd in commands:
    logger.info(f"Running command: {cmd}")
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        logger.info(f"Output:\n{result.stdout}")
        if result.stderr:
            logger.error(f"Errors:\n{result.stderr}")
        if result.returncode != 0:
            logger.error(f"Command failed with return code {result.returncode}")
            break
    except Exception as e:
        logger.error(f"Failed to run command: {e}")
        break
