"""
=========================================================
Visualize Predicted Heatmaps

Saves all predicted heatmaps as images.

=========================================================
"""

import os
import cv2
import numpy as np

from inference import predict
from configs.config import Config

cfg = Config

# --------------------------------------------------------
# Main
# --------------------------------------------------------

if __name__ == "__main__":

    image_path = os.path.join(
        cfg.MPII_IMAGE_DIR,
        "037454012.jpg"
    )

    image, heatmaps, joints = predict(image_path)

    save_dir = os.path.join(
        cfg.OUTPUT_DIR,
        "heatmaps"
    )

    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Saving Heatmaps...")
    print("=" * 60)

    for i in range(cfg.NUM_JOINTS):

        hm = heatmaps[i]

        print(
            f"Joint {i:02d} | "
            f"Max={hm.max():.4f} "
            f"Min={hm.min():.4f} "
            f"Mean={hm.mean():.6f}"
        )

        hm = hm - hm.min()

        if hm.max() > 0:
            hm = hm / hm.max()

        hm = (hm * 255).astype(np.uint8)

        hm = cv2.applyColorMap(
            hm,
            cv2.COLORMAP_JET
        )

        cv2.imwrite(
            os.path.join(
                save_dir,
                f"heatmap_{i}.png"
            ),
            hm
        )

    print("\nSaved to:")
    print(save_dir)