import torch

from models.vit_pose import ViTPose


model = ViTPose()


images = torch.randn(
    2,
    3,
    256,
    256
)


output = model(images)


print("Input:")
print(images.shape)


print("\nPredicted Heatmaps:")
print(output.shape)