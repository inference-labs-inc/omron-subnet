import torch
import json
from baseline_model import MatMulModel


def print_matrix(tensor, shape):
    matrix = tensor.reshape(shape)
    for row in matrix:
        print([f"{x:.1f}" for x in row])


with open("input.json", "r") as f:
    data = json.load(f)
    input_tensor = torch.tensor(data["input_data"][0], dtype=torch.float32).unsqueeze(0)

print("Matrix 1 (2x5):")
print_matrix(input_tensor[0, :10], (2, 5))

print("\nMatrix 2 (5x3):")
print_matrix(input_tensor[0, 10:], (5, 3))

model = MatMulModel()
with torch.no_grad():
    output = model(input_tensor)

print("\nResult (2x3):")
print_matrix(output[0], (2, 3))
