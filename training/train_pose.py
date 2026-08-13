"""
=========================================================
Research-Oriented ViT Pose Training

MPII Human Pose Estimation

Author : Anil
=========================================================
"""

import os
import sys
import time
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split

# ---------------------------------------------------
# Add project root
# ---------------------------------------------------

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if ROOT not in sys.path:
    sys.path.append(ROOT)

# ---------------------------------------------------
# Project Imports
# ---------------------------------------------------

from datasets.mpii_train_dataset import MPIITrainDataset
from preprocessing.transforms import train_transform
from models.vit_pose import ViTPose

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

IMAGE_DIR = "datasets/mpii/images"

ANNOTATION_FILE = (
    "datasets/mpii/annotations/"
    "mpii_human_pose_v1_u12_1.mat"
)

CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 1e-4

TRAIN_SPLIT = 0.80

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("Device :", DEVICE)
print("=" * 60)

# ---------------------------------------------------
# Dataset
# ---------------------------------------------------

dataset = MPIITrainDataset(
    image_dir=IMAGE_DIR,
    annotation_file=ANNOTATION_FILE,
    transform=train_transform
)

train_size = int(
    TRAIN_SPLIT * len(dataset)
)

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print("Training Samples :", len(train_dataset))
print("Validation Samples :", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

# ---------------------------------------------------
# Model
# ---------------------------------------------------

model = ViTPose(
    image_size=256,
    patch_size=16,
    embed_dim=256,
    depth=4,
    num_heads=8,
    num_joints=16
).to(DEVICE)

print("\nModel Loaded Successfully\n")

# ---------------------------------------------------
# Loss
# ---------------------------------------------------

criterion = nn.MSELoss()

# ---------------------------------------------------
# Optimizer
# ---------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

best_loss = float("inf")
start_epoch = 0
# =====================================================
# Resume Training (Optional)
# =====================================================

RESUME = False

RESUME_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

if RESUME and os.path.exists(RESUME_PATH):

    checkpoint = torch.load(
        RESUME_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    best_loss = checkpoint["best_loss"]

    print("=" * 60)
    print("Checkpoint Loaded")
    print("Resuming from Epoch:", start_epoch)
    print("=" * 60)


# =====================================================
# Training Loop
# =====================================================

for epoch in range(start_epoch, EPOCHS):

    print("\n")
    print("=" * 70)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("=" * 70)

    epoch_start = time.time()

    model.train()

    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(train_loader):

        images = images.to(DEVICE)

        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(
            predictions,
            targets
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:

            print(
                f"Batch "
                f"{batch_idx:4d}/{len(train_loader)}"
                f" | Loss : {loss.item():.6f}"
            )

    train_loss = (
        running_loss /
        len(train_loader)
    )

    scheduler.step()

    print("\nTraining Loss :", train_loss)
        # =====================================================
    # Validation
    # =====================================================

    model.eval()

    validation_loss = 0.0

    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            predictions = model(images)

            loss = criterion(
                predictions,
                targets
            )

            validation_loss += loss.item()

    validation_loss /= len(val_loader)

    print(
        f"Validation Loss : {validation_loss:.6f}"
    )

    # =====================================================
    # Save Best Model
    # =====================================================

    if validation_loss < best_loss:

        best_loss = validation_loss

        best_path = os.path.join(
            CHECKPOINT_DIR,
            "best_model.pth"
        )

        torch.save({

            "epoch": epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "best_loss":
                best_loss

        }, best_path)

        print("\nBest model updated!")

    # =====================================================
    # Save Epoch Checkpoint
    # =====================================================

    checkpoint_path = os.path.join(

        CHECKPOINT_DIR,

        f"vit_pose_epoch_{epoch+1}.pth"

    )

    torch.save({

        "epoch": epoch + 1,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "validation_loss":
            validation_loss

    }, checkpoint_path)

    print(
        f"Checkpoint Saved : {checkpoint_path}"
    )

    # =====================================================
    # Epoch Summary
    # =====================================================

    epoch_time = time.time() - epoch_start

    print("\n--------------------------------------")
    print(f"Epoch {epoch+1} Completed")
    print(f"Training Loss   : {train_loss:.6f}")
    print(f"Validation Loss : {validation_loss:.6f}")
    print(f"Best Loss       : {best_loss:.6f}")
    print(f"Time            : {epoch_time/60:.2f} min")
    print("--------------------------------------")


# =====================================================
# Training Finished
# =====================================================

print("\n")
print("=" * 70)
print("Training Completed Successfully!")
print("=" * 70)

print("\nBest Validation Loss :", best_loss)
print("Best Model :", os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
))