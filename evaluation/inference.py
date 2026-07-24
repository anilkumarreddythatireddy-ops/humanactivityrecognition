"""
=========================================================
ViTPose Inference

Loads trained model and predicts joint heatmaps.

Author : Anil
=========================================================
"""

import os
import sys
import cv2
import torch
import numpy as np

# =========================================================
# Add Project Root
# =========================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if ROOT not in sys.path:
    sys.path.append(ROOT)

# =========================================================
# Project Imports
# =========================================================

from models.vit_pose import ViTPose
from preprocessing.transforms import train_transform
from configs.config import Config

cfg = Config

# =========================================================
# Device
# =========================================================

DEVICE = cfg.DEVICE

# =========================================================
# Load Model
# =========================================================

model = ViTPose(
    image_size=cfg.IMAGE_SIZE,
    patch_size=cfg.PATCH_SIZE,
    embed_dim=cfg.EMBED_DIM,
    depth=cfg.NUM_LAYERS,
    num_heads=cfg.NUM_HEADS,
    num_joints=cfg.NUM_JOINTS
)

checkpoint_path = os.path.join(
    cfg.CHECKPOINT_DIR,
    "best_model.pth"
)

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{checkpoint_path}"
    )

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)

model.eval()

print("=" * 60)
print("Model Loaded Successfully")
print("Checkpoint :", checkpoint_path)
print("Device :", DEVICE)
print("=" * 60)


# =========================================================
# Heatmap -> Joint Coordinates
# =========================================================

def heatmaps_to_joints(heatmaps):

    joints = []

    scale = cfg.IMAGE_SIZE // cfg.HEATMAP_SIZE

    for hm in heatmaps:

        y, x = np.unravel_index(
            np.argmax(hm),
            hm.shape
        )

        confidence = float(hm[y, x])

        joints.append(

            {
                "x": int(x * scale),
                "y": int(y * scale),
                "confidence": confidence
            }

        )

    return joints


# =========================================================
# Predict
# =========================================================

def predict(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            image_path
        )

    original = image.copy()

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (
            cfg.IMAGE_SIZE,
            cfg.IMAGE_SIZE
        )
    )

    tensor = train_transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        heatmaps = model(tensor)

    heatmaps = heatmaps.squeeze(0)

    heatmaps = heatmaps.cpu().numpy()

    joints = heatmaps_to_joints(
        heatmaps
    )

    return original, heatmaps, joints


# =========================================================
# Print Joints
# =========================================================

def print_joints(joints):

    print()

    print("=" * 60)
    print("Predicted Joint Coordinates")
    print("=" * 60)

    for idx, joint in enumerate(joints):

        print(

            f"Joint {idx:02d}"

            f" -> "

            f"X={joint['x']:3d}"

            f"  "

            f"Y={joint['y']:3d}"

            f"  "

            f"Confidence={joint['confidence']:.4f}"

        )


# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

    image_path = os.path.join(
        cfg.MPII_IMAGE_DIR,
        "037454012.jpg"
    )

    image, heatmaps, joints = predict(
        image_path
    )

    print_joints(joints)

    print()

    print("Heatmap Shape :", heatmaps.shape)

    print("Done.")