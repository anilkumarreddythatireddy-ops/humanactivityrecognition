"""
=========================================================
Learnable Positional Encoding

Vision Transformer

Research Project:
Human Pose Estimation and Activity Recognition
=========================================================
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Learnable positional embeddings.

    Input:
        (B, N, D)

    Output:
        (B, N, D)
    """

    def __init__(
        self,
        num_patches=256,
        embed_dim=768,
        dropout=0.1,
    ):

        super().__init__()

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                num_patches,
                embed_dim
            )
        )

        self.dropout = nn.Dropout(dropout)

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02
        )

    def forward(self, x):

        x = x + self.position_embedding

        x = self.dropout(x)

        return x