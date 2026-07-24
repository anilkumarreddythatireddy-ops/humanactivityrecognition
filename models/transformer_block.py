"""
=========================================================
Transformer Encoder Block

Vision Transformer

Research Project:
Human Pose Estimation and Activity Recognition
=========================================================
"""

import torch.nn as nn

from models.attention import MultiHeadSelfAttention
from models.mlp import MLP



class TransformerBlock(nn.Module):

    def __init__(
            self,
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4,
            dropout=0.1):

        super().__init__()


        # Layer Normalization

        self.norm1 = nn.LayerNorm(
            embed_dim
        )


        self.attention = MultiHeadSelfAttention(
            embed_dim,
            num_heads,
            dropout
        )


        self.norm2 = nn.LayerNorm(
            embed_dim
        )


        self.mlp = MLP(
            embed_dim,
            mlp_ratio,
            dropout
        )


    def forward(self, x):

        # Attention + Residual

        x = x + self.attention(
            self.norm1(x)
        )


        # MLP + Residual

        x = x + self.mlp(
            self.norm2(x)
        )


        return x