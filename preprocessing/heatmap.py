"""
=========================================================
Heatmap Generator for MPII Human Pose Estimation
=========================================================
"""

import numpy as np


class HeatmapGenerator:
    """
    Generate Gaussian heatmaps for human joints.
    """

    def __init__(
        self,
        image_size=256,
        heatmap_size=64,
        num_joints=16,
        sigma=2,
    ):

        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.num_joints = num_joints
        self.sigma = sigma

        self.stride = image_size / heatmap_size

    def _gaussian2D(self, shape, sigma):

        m = (shape[0] - 1) / 2
        n = (shape[1] - 1) / 2

        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

        h[h < np.finfo(h.dtype).eps * h.max()] = 0

        return h

    def generate(self, joints, visibility):

        heatmaps = np.zeros(
            (
                self.num_joints,
                self.heatmap_size,
                self.heatmap_size,
            ),
            dtype=np.float32,
        )

        tmp_size = self.sigma * 3

        gaussian = self._gaussian2D(
            (2 * tmp_size + 1, 2 * tmp_size + 1),
            self.sigma,
        )

        for joint in range(self.num_joints):

            if visibility[joint] <= 0:
                continue

            x = joints[joint][0] / self.stride
            y = joints[joint][1] / self.stride

            mu_x = int(round(x))
            mu_y = int(round(y))

            # completely outside heatmap
            if (
                mu_x < 0
                or mu_y < 0
                or mu_x >= self.heatmap_size
                or mu_y >= self.heatmap_size
            ):
                continue

            left = min(tmp_size, mu_x)
            right = min(tmp_size, self.heatmap_size - mu_x - 1)

            top = min(tmp_size, mu_y)
            bottom = min(tmp_size, self.heatmap_size - mu_y - 1)

            heatmaps[
                joint,
                mu_y - top:mu_y + bottom + 1,
                mu_x - left:mu_x + right + 1,
            ] = gaussian[
                tmp_size - top:tmp_size + bottom + 1,
                tmp_size - left:tmp_size + right + 1,
            ]

        return heatmaps