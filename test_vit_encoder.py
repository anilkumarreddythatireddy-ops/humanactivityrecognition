import torch

from models.vit_encoder import ViTEncoder


model = ViTEncoder(
    depth=12
)


x = torch.randn(
    2,
    3,
    256,
    256
)


features = model(x)


print("Input:", x.shape)

print("ViT Output:", features.shape)