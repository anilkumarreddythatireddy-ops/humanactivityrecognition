"""
=========================================================
Multi-Head Self Attention

Vision Transformer

Research Project:
Human Pose Estimation and Activity Recognition
=========================================================
"""

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    """
    Multi-Head Self Attention

    Input:
        (B, N, D)

    Output:
        (B, N, D)

    B = batch size
    N = number of tokens
    D = embedding dimension
    """

    def __init__(
            self,
            embed_dim=768,
            num_heads=12,
            dropout=0.1):

        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.head_dim = (
            embed_dim // num_heads
        )

        self.scale = (
            self.head_dim ** -0.5
        )


        # Q,K,V projection

        self.qkv = nn.Linear(
            embed_dim,
            embed_dim * 3
        )


        self.attn_dropout = nn.Dropout(
            dropout
        )


        self.projection = nn.Linear(
            embed_dim,
            embed_dim
        )


        self.proj_dropout = nn.Dropout(
            dropout
        )


    def forward(self, x):

        B, N, C = x.shape


        # Generate Q,K,V

        qkv = self.qkv(x)


        qkv = qkv.reshape(
            B,
            N,
            3,
            self.num_heads,
            self.head_dim
        )


        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4
        )


        q, k, v = qkv[0], qkv[1], qkv[2]


        # Attention score

        attention = (
            q @ k.transpose(-2,-1)
        )


        attention = (
            attention * self.scale
        )


        attention = torch.softmax(
            attention,
            dim=-1
        )


        attention = self.attn_dropout(
            attention
        )


        # Weighted sum

        output = (
            attention @ v
        )


        output = output.transpose(
            1,
            2
        )


        output = output.reshape(
            B,
            N,
            C
        )


        output = self.projection(
            output
        )


        output = self.proj_dropout(
            output
        )


        return output