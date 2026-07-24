"""
=============================================================
MPII Human Pose Dataset Loader

Research Project:
Human Pose Estimation and Activity Recognition
using Vision Transformers and Variational Autoencoders

Author: T. Anil Kumar Reddy
=============================================================
"""

import os
import cv2
import numpy as np

from scipy.io import loadmat

import torch
from torch.utils.data import Dataset


class MPIIDataset(Dataset):

    NUM_JOINTS = 16

    def __init__(
            self,
            image_dir,
            annotation_file,
            transform=None,
            train_only=True):

        self.image_dir = image_dir
        self.transform = transform
        self.train_only = train_only

        print("=" * 60)
        print("Loading MPII Dataset...")
        print("=" * 60)

        mat = loadmat(
            annotation_file,
            struct_as_record=False,
            squeeze_me=True
        )

        self.release = mat["RELEASE"]

        self.annolist = self.release.annolist
        self.img_train = self.release.img_train
        self.single_person = self.release.single_person

        self.samples = []

        self._prepare_dataset()

        print("=" * 60)
        print("Finished Loading Dataset")
        print("Total Samples :", len(self.samples))
        print("=" * 60)

    def __len__(self):
        return len(self.samples)

    def _empty_joints(self):

        joints = np.zeros(
            (self.NUM_JOINTS, 2),
            dtype=np.float32
        )

        visibility = np.zeros(
            self.NUM_JOINTS,
            dtype=np.float32
        )

        return joints, visibility

    def _prepare_dataset(self):

        total = len(self.annolist)

        print("Parsing annotations...")

        for idx in range(total):

            if self.train_only:

                if self.img_train[idx] != 1:
                    continue

            ann = self.annolist[idx]

            if not hasattr(ann, "annorect"):
                continue

            annorect = ann.annorect

            if annorect is None:
                continue

            # Handle one person or multiple people
            if not isinstance(annorect, (list, tuple, np.ndarray)):
                annorect = [annorect]

            for person_index, person in enumerate(annorect):

                if not hasattr(person, "annopoints"):
                    continue

                if person.annopoints is None:
                    continue

                joints, visibility = self._extract_joints(
                    person.annopoints
                )

                sample = {

                    "image_name": ann.image.name,

                    "joints": joints,

                    "visibility": visibility,

                    "center": self._get_center(person),

                    "scale": self._get_scale(person),

                    "person_index": person_index
                }

                self.samples.append(sample)

    def _extract_joints(self, annopoints):

        joints, visibility = self._empty_joints()

        if not hasattr(annopoints, "point"):
            return joints, visibility

        points = annopoints.point

        if not isinstance(points, (list, tuple, np.ndarray)):
            points = [points]

        for p in points:

            joint_id = int(p.id)

            if joint_id >= self.NUM_JOINTS:
                continue

            joints[joint_id] = [

                float(p.x),

                float(p.y)

            ]

            if hasattr(p, "is_visible"):

                try:
                    visibility[joint_id] = int(p.is_visible)
                except:
                    visibility[joint_id] = 1

            else:

                visibility[joint_id] = 1

        return joints, visibility
    def _get_center(self, person):
    
        if not hasattr(person, "objpos"):
            return (0.0, 0.0)

        objpos = person.objpos

        if objpos is None:
            return (0.0, 0.0)

    # Handle ndarray
        if isinstance(objpos, np.ndarray):
            if len(objpos) == 0:
                return (0.0, 0.0)
            objpos = objpos.flat[0]

        try:
            return (
                float(objpos.x),
                float(objpos.y)
            )
        except Exception:
            return (0.0, 0.0)

    def _get_scale(self, person):
    

        if not hasattr(person, "scale"):
            return 1.0

        scale = person.scale

        if scale is None:
            return 1.0

        if isinstance(scale, np.ndarray):
            if len(scale) == 0:
                return 1.0
            scale = scale.flat[0]

        try:
            return float(scale)
        except Exception:
            return 1.0

    def __getitem__(self, index):

        sample = self.samples[index]

        image_path = os.path.join(
            self.image_dir,
            sample["image_name"]
        )

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ORIGINAL SIZE
        orig_h, orig_w = image.shape[:2]

        # Compute scaling factors BEFORE resize
        scale_x = 256.0 / orig_w
        scale_y = 256.0 / orig_h

        scaled_joints = sample["joints"].copy()

        scaled_joints[:, 0] *= scale_x
        scaled_joints[:, 1] *= scale_y

        # Now transform image
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)

        joints = torch.tensor(
            scaled_joints,
            dtype=torch.float32
        )

        visibility = torch.tensor(
            sample["visibility"],
            dtype=torch.float32
        )

        metadata = {
            "image_name": sample["image_name"],
            "center": sample["center"],
            "scale": sample["scale"],
            "person_index": sample["person_index"]
        }

        return image, joints, visibility, metadata