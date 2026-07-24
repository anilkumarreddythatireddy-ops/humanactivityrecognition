"""
=========================================================
MPII Training Dataset

Image + Ground Truth Heatmaps

For ViT Pose Training

=========================================================
"""

import torch
from torch.utils.data import Dataset

from preprocessing.mpii_dataset import MPIIDataset
from preprocessing.heatmap import HeatmapGenerator



class MPIITrainDataset(Dataset):

    def __init__(
            self,
            image_dir,
            annotation_file,
            transform=None):

        self.dataset = MPIIDataset(
            image_dir=image_dir,
            annotation_file=annotation_file,
            transform=transform
        )


        self.heatmap_generator = HeatmapGenerator(
            image_size=256,
            heatmap_size=64,
            num_joints=16,
            sigma=2
        )


    def __len__(self):

        return len(self.dataset)


    def __getitem__(self, index):

        image, joints, visibility, meta = self.dataset[index]


        heatmaps = self.heatmap_generator.generate(
            joints.numpy(),
            visibility.numpy()
        )


        heatmaps = torch.tensor(
            heatmaps,
            dtype=torch.float32
        )


        return image, heatmaps