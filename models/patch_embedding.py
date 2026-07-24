"""
=========================================================
Patch Embedding

Vision Transformer

Research Project:
Human Pose Estimation and Activity Recognition
=========================================================
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Convert image into patch tokens.

    Input:
        (B,3,256,256)

    Output:
        (B,256,768)

    256 patches
    768-dimensional embedding
    """

    def __init__(
            self,
            image_size=256,
            patch_size=16,
            in_channels=3,
            embed_dim=768):

        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patches = (
            image_size // patch_size
        ) ** 2

        self.projection = nn.Conv2d(

            in_channels,

            embed_dim,

            kernel_size=patch_size,

            stride=patch_size

        )

    def forward(self, x):

        x = self.projection(x)

        # B,C,H,W
        # B,768,16,16

        x = x.flatten(2)

        # B,768,256

        x = x.transpose(1, 2)

        # B,256,768

        return x