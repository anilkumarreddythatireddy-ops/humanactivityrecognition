import torch

from models.pose_decoder import PoseDecoder

decoder = PoseDecoder()

x = torch.randn(2,256,768)

heatmaps = decoder(x)

print("Input :",x.shape)
print("Output:",heatmaps.shape)