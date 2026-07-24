import torch
import torch.nn as nn


class PoseDecoder(nn.Module):

    def __init__(
        self,
        embed_dim=768,
        num_joints=16
    ):

        super().__init__()

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                embed_dim,
                256,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                num_joints,
                kernel_size=1
            )

        )

    def forward(self, x):

        B, N, C = x.shape

        H = W = int(N ** 0.5)

        x = x.transpose(1, 2)

        x = x.reshape(B, C, H, W)

        heatmaps = self.decoder(x)

        return heatmaps