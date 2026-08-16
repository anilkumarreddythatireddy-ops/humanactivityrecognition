import torch

from models.mlp import MLP


mlp = MLP()


x = torch.randn(
    2,
    256,
    768
)


output = mlp(x)


print("Input :", x.shape)

print("Output:", output.shape)