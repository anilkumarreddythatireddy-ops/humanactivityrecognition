import torch

from models.attention import MultiHeadSelfAttention


attention = MultiHeadSelfAttention()


x = torch.randn(
    2,
    256,
    768
)


output = attention(x)


print("Input:", x.shape)

print("Output:", output.shape)