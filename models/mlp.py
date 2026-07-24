"""
=========================================================
Transformer MLP / Feed Forward Network

Vision Transformer

Research Project:
Human Pose Estimation and Activity Recognition
=========================================================
"""

import torch.nn as nn


class MLP(nn.Module):

    """
    Feed Forward Network used inside Transformer.

    Input:
        (B,N,D)

    Output:
        (B,N,D)
    """

    def __init__(
            self,
            embed_dim=768,
            mlp_ratio=4,
            dropout=0.1):

        super().__init__()

        hidden_dim = int(
            embed_dim * mlp_ratio
        )


        self.network = nn.Sequential(

            nn.Linear(
                embed_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(dropout),


            nn.Linear(
                hidden_dim,
                embed_dim
            ),

            nn.Dropout(dropout)

        )


    def forward(self, x):

        return self.network(x)