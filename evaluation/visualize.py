"""
=========================================================
ViTPose Skeleton Visualization

Loads predicted joints and draws the MPII skeleton.

=========================================================
"""

import os
import cv2

from inference import predict
from configs.config import Config

cfg = Config

# --------------------------------------------------------
# MPII Skeleton Connections
# --------------------------------------------------------

SKELETON = [

    (0, 1),
    (1, 2),

    (3, 4),
    (4, 5),

    (2, 6),
    (3, 6),

    (6, 7),

    (7, 8),

    (8, 9),

    (7, 12),

    (10, 11),
    (11, 12),

    (12, 13),

    (13, 14),

    (14, 15)

]

# --------------------------------------------------------
# Draw Skeleton
# --------------------------------------------------------

def draw_pose(image, joints):

    image = image.copy()

    # Draw joints

    for joint in joints:

        x = joint["x"]
        y = joint["y"]

        cv2.circle(
            image,
            (x, y),
            4,
            (0, 0, 255),
            -1
        )

    # Draw bones

    for start, end in SKELETON:

        x1 = joints[start]["x"]
        y1 = joints[start]["y"]

        x2 = joints[end]["x"]
        y2 = joints[end]["y"]

        cv2.line(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    return image


# --------------------------------------------------------
# Main
# --------------------------------------------------------

if __name__ == "__main__":

    image_path = os.path.join(
        cfg.MPII_IMAGE_DIR,
        "037454012.jpg"
    )

    image, heatmaps, joints = predict(image_path)

    result = draw_pose(image, joints)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    save_path = os.path.join(
        cfg.OUTPUT_DIR,
        "prediction.jpg"
    )

    cv2.imwrite(save_path, result)

    print("=" * 60)
    print("Prediction saved to:")
    print(save_path)
    print("=" * 60)

    cv2.imshow("Prediction", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()