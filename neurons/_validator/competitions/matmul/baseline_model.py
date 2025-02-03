import torch
import torch.nn as nn


class MatMulModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        matrix1 = x[:, :10].reshape(batch_size, 2, 5)
        matrix2 = x[:, 10:].reshape(batch_size, 5, 3)
        return torch.matmul(matrix1, matrix2).reshape(batch_size, -1)


if __name__ == "__main__":
    model = MatMulModel()
    torch.save(model, "matmul_baseline.pt")
