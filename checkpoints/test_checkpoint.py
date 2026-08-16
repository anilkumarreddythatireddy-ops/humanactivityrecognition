import torch

checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location="cpu"
)

print(checkpoint.keys())

print("\nEpoch:", checkpoint["epoch"])

print("Best Loss:", checkpoint["best_loss"])