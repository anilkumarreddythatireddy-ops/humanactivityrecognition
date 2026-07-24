"""
==========================================================
Project:
Human Pose Estimation and Activity Recognition
using Vision Transformers and Variational Autoencoders

Author:
T. Anil Kumar Reddy

Description:
Global configuration file.
==========================================================
"""

import os
import random
import numpy as np
import torch


class Config:

    # ======================================================
    # Dataset Paths
    # ======================================================

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    MPII_IMAGE_DIR = os.path.join(
        PROJECT_ROOT,
        "datasets",
        "mpii",
        "images"
    )

    MPII_ANNOTATION_FILE = os.path.join(
        PROJECT_ROOT,
        "datasets",
        "mpii",
        "annotations",
        "mpii_human_pose_v1_u12_1.mat"
    )

    UCF101_DIR = os.path.join(
        PROJECT_ROOT,
        "datasets",
        "ucf101"
    )

    # ======================================================
    # Image Parameters
    # ======================================================

    IMAGE_SIZE = 256
    HEATMAP_SIZE = 64
    NUM_CHANNELS = 3

    # ======================================================
    # MPII Parameters
    # ======================================================

    NUM_JOINTS = 16

    # ======================================================
    # Vision Transformer
    # ======================================================

    # ======================================================
    # Vision Transformer
    # ======================================================

    PATCH_SIZE = 16

    EMBED_DIM = 768

    NUM_HEADS = 12

    NUM_LAYERS = 12

    MLP_RATIO = 4

    DROPOUT = 0.1

    # ======================================================
    # Pose Decoder
    # ======================================================

    DECODER_CHANNELS = 256

    # ======================================================
    # Activity Recognition
    # ======================================================

    NUM_CLASSES = 101

    SEQUENCE_LENGTH = 16

    LATENT_DIM = 128
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ======================================================
    # Training
    # ======================================================

    BATCH_SIZE = 16

    LEARNING_RATE = 1e-4

    WEIGHT_DECAY = 1e-4

    POSE_EPOCHS = 50

    ACTIVITY_EPOCHS = 60

    NUM_WORKERS = 2

    PIN_MEMORY = True
    # ======================================================
    # Loss Weights
    # ======================================================

    LAMBDA_POSE = 1.0

    LAMBDA_CLASS = 1.0

    LAMBDA_REC = 0.2

    LAMBDA_KL = 0.001

    # ======================================================
    # Device
    # ======================================================

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ======================================================
    # Output Directories
    # ======================================================

    CHECKPOINT_DIR = os.path.join(
        PROJECT_ROOT,
        "checkpoints"
    )

    LOG_DIR = os.path.join(
        PROJECT_ROOT,
        "logs"
    )

    OUTPUT_DIR = os.path.join(
        PROJECT_ROOT,
        "outputs"
    )


def set_seed(seed=42):
    """
    Make experiments reproducible.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False