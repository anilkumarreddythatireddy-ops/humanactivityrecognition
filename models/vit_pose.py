"""
=========================================================
ViT Pose Estimation Model
=========================================================
"""

import torch.nn as nn

from configs.config import Config
from models.vit_encoder import ViTEncoder
from models.pose_decoder import PoseDecoder

cfg = Config


class ViTPose(nn.Module):

    def __init__(self, num_joints=cfg.NUM_JOINTS):

        super().__init__()

        self.encoder = ViTEncoder(
            image_size=cfg.IMAGE_SIZE,
            patch_size=cfg.PATCH_SIZE,
            embed_dim=cfg.EMBED_DIM,
            depth=cfg.NUM_LAYERS,
            num_heads=cfg.NUM_HEADS
        )

        self.decoder = PoseDecoder(
            embed_dim=cfg.EMBED_DIM,
            num_joints=num_joints
        )

    def forward(self, x):

        features = self.encoder(x)

        heatmaps = self.decoder(features)

        return heatmaps