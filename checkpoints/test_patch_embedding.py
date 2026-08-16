import torch

from models.patch_embedding import PatchEmbedding

model = PatchEmbedding()

x = torch.randn(

    2,

    3,

    256,

    256

)

y = model(x)

print("Input :", x.shape)

print("Output:", y.shape)

print("Number of patches:", model.num_patches)