"""
=========================================================
Vision Transformer Encoder

Pose Estimation Branch

Architecture:

Image
 |
Patch Embedding
 |
Position Embedding
 |
Transformer Encoder Blocks
 |
Feature Tokens

=========================================================
"""

import torch.nn as nn

from models.patch_embedding import PatchEmbedding
from models.positional_encoding import PositionalEncoding
from models.transformer_block import TransformerBlock



class ViTEncoder(nn.Module):

    def __init__(
            self,
            image_size=256,
            patch_size=16,
            embed_dim=268,
            depth=12,
            num_heads=12,
            dropout=0.1):

        super().__init__()


        # Patch extraction

        self.patch_embedding = PatchEmbedding(
            image_size,
            patch_size,
            3,
            embed_dim
        )


        num_patches = (
            image_size // patch_size
        ) ** 2


        # Position information

        self.position_embedding = PositionalEncoding(
            num_patches,
            embed_dim,
            dropout
        )


        # Transformer Encoder

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    4,
                    dropout
                )

                for _ in range(depth)
            ]
        )


        self.norm = nn.LayerNorm(
            embed_dim
        )


    def forward(self, x):

        # Image → Patch tokens

        x = self.patch_embedding(x)


        # Add position

        x = self.position_embedding(x)


        # Transformer blocks

        for block in self.blocks:

            x = block(x)


        x = self.norm(x)


        return x