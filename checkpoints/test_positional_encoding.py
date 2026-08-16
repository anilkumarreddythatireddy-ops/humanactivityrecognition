import torch

from models.patch_embedding import PatchEmbedding
from models.positional_encoding import PositionalEncoding

patch = PatchEmbedding()

position = PositionalEncoding()

x = torch.randn(
    2,
    3,
    256,
    256
)

tokens = patch(x)

output = position(tokens)

print("Patch Tokens :", tokens.shape)
print("Position Encoded :", output.shape)