import torch

from models.transformer_block import TransformerBlock


block = TransformerBlock()


x = torch.randn(
    2,
    256,
    768
)


output = block(x)


print("Input :", x.shape)

print("Output:", output.shape) 