import torch

model = torch.load("matmul_baseline.pt")
model.eval()

dummy_input = torch.randn(1, 25)
torch.onnx.export(
    model,
    dummy_input,
    "baseline.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
