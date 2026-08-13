from pathlib import Path
from textwrap import dedent

code = dedent(r'''
"""
=========================================================
ViT Pose Training - MPII
=========================================================

Features
--------
- MPII train/validation split
- ViTPose model matching the current project architecture
- MSE heatmap loss
- PCK-style heatmap evaluation
- Automatic resume from last_checkpoint.pth
- Saves a checkpoint after EVERY epoch
- Saves best_model.pth based on validation PCK
- Saves best_loss_model.pth based on validation loss
- CUDA / Kaggle GPU support
- Mixed precision on CUDA
- Gradient clipping
- Fixed random split for reproducibility

IMPORTANT
---------
No training script can honestly guarantee 60-85% PCK/accuracy before
the dataset, heatmap generation, coordinate scaling, model and metric
are validated. This script is designed to improve and measure the model;
the actual score depends on those components.
"""

import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

# =========================================================
# Project root
# =========================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# =========================================================
# Project imports
# =========================================================

from datasets.mpii_train_dataset import MPIITrainDataset
from preprocessing.transforms import train_transform
from models.vit_pose import ViTPose

# =========================================================
# Configuration
# =========================================================

IMAGE_DIR = os.path.join(
    ROOT, "datasets", "mpii", "images"
)

ANNOTATION_FILE = os.path.join(
    ROOT,
    "datasets",
    "mpii",
    "annotations",
    "mpii_human_pose_v1_u12_1.mat"
)

CHECKPOINT_DIR = os.path.join(
    ROOT, "checkpoints"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Training settings
# ---------------------------------------------------------

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

TRAIN_SPLIT = 0.80

# PCK threshold in heatmap pixels.
# 5/64 pixels is a reasonably strict threshold.
PCK_THRESHOLD = 5.0

# Set to 0 if Kaggle has worker issues.
NUM_WORKERS = 2

# =========================================================
# Device
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"

print("=" * 70)
print("ViTPose MPII Training")
print("=" * 70)
print("Device       :", DEVICE)
print("AMP          :", USE_AMP)
print("Batch Size   :", BATCH_SIZE)
print("Epochs       :", EPOCHS)
print("Learning Rate:", LEARNING_RATE)
print("=" * 70)

if DEVICE.type == "cuda":
    print("GPU          :", torch.cuda.get_device_name(0))
    print("CUDA Memory  :",
          round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
          "GB")
    print("=" * 70)

# =========================================================
# Reproducibility
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================================================
# Dataset
# =========================================================

print("\nLoading MPII dataset...")

dataset = MPIITrainDataset(
    image_dir=IMAGE_DIR,
    annotation_file=ANNOTATION_FILE,
    transform=train_transform
)

train_size = int(
    TRAIN_SPLIT * len(dataset)
)

val_size = len(dataset) - train_size

split_generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=split_generator
)

print("Total Samples     :", len(dataset))
print("Training Samples  :", len(train_dataset))
print("Validation Samples:", len(val_dataset))

# =========================================================
# Data loaders
# =========================================================

pin_memory = DEVICE.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
    persistent_workers=(NUM_WORKERS > 0)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
    persistent_workers=(NUM_WORKERS > 0)
)

print("Training batches  :", len(train_loader))
print("Validation batches:", len(val_loader))

# =========================================================
# Model
# =========================================================
#
# This matches the smaller ViTPose architecture used by the
# current train_pose.py:
#
# image_size = 256
# patch_size = 16
# embed_dim  = 256
# depth      = 4
# heads      = 8
# joints     = 16
#
# Do NOT change these values if you want to resume a
# checkpoint created with this architecture.
# =========================================================

model = ViTPose(
    image_size=256,
    patch_size=16,
    embed_dim=256,
    depth=4,
    num_heads=8,
    num_joints=16
).to(DEVICE)

print("\nModel created successfully.")

# =========================================================
# Loss
# =========================================================

criterion = nn.MSELoss()

# =========================================================
# Optimizer
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# =========================================================
# Scheduler
# =========================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)

# =========================================================
# AMP scaler
# =========================================================

try:
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(
        enabled=USE_AMP
    )

# =========================================================
# Checkpoint paths
# =========================================================

LAST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "last_checkpoint.pth"
)

BEST_MODEL = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

BEST_LOSS_MODEL = os.path.join(
    CHECKPOINT_DIR,
    "best_loss_model.pth"
)

# =========================================================
# Resume state
# =========================================================

start_epoch = 0
best_pck = 0.0
best_val_loss = float("inf")


def load_checkpoint(path):
    global start_epoch
    global best_pck
    global best_val_loss

    print("\n" + "=" * 70)
    print("Loading checkpoint")
    print(path)
    print("=" * 70)

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    if "scaler_state_dict" in checkpoint:
        try:
            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )
        except Exception:
            print("Warning: AMP scaler state could not be restored.")

    # Checkpoints saved by this script store the completed
    # epoch as epoch_number. Older project checkpoints used
    # different conventions, so handle both.
    if "epoch_number" in checkpoint:
        completed_epoch = int(
            checkpoint["epoch_number"]
        )
        start_epoch = completed_epoch

    elif "epoch" in checkpoint:
        completed_epoch = int(
            checkpoint["epoch"]
        )

        # Existing project checkpoints store epoch+1.
        # Treat it as the next epoch when it is within range.
        start_epoch = completed_epoch

    else:
        start_epoch = 0

    best_pck = float(
        checkpoint.get(
            "best_pck",
            0.0
        )
    )

    best_val_loss = float(
        checkpoint.get(
            "best_val_loss",
            checkpoint.get(
                "best_loss",
                float("inf")
            )
        )
    )

    print("Completed epoch :", start_epoch)
    print("Next epoch      :", start_epoch + 1)
    print("Best PCK        :", f"{best_pck:.2f}%")
    print("Best Val Loss   :", f"{best_val_loss:.8f}")
    print("=" * 70)


# =========================================================
# Automatically resume from last checkpoint
# =========================================================

if os.path.exists(LAST_CHECKPOINT):

    load_checkpoint(
        LAST_CHECKPOINT
    )

else:

    print("\nNo last_checkpoint.pth found.")
    print("Starting from epoch 1.")

# =========================================================
# Heatmap PCK
# =========================================================

def heatmap_pck(
    predictions,
    targets,
    threshold=PCK_THRESHOLD
):
    """
    PCK-style metric computed from heatmap argmax positions.

    This is a heatmap-space diagnostic metric, not official
    MPII PCKh. Official PCKh requires the original joint
    coordinates and head-size normalization.

    predictions: [B, J, H, W]
    targets    : [B, J, H, W]
    """

    with torch.no_grad():

        pred_flat = predictions.detach().reshape(
            predictions.shape[0],
            predictions.shape[1],
            -1
        )

        target_flat = targets.detach().reshape(
            targets.shape[0],
            targets.shape[1],
            -1
        )

        pred_indices = pred_flat.argmax(
            dim=-1
        )

        target_indices = target_flat.argmax(
            dim=-1
        )

        width = predictions.shape[-1]

        pred_y = pred_indices // width
        pred_x = pred_indices % width

        target_y = target_indices // width
        target_x = target_indices % width

        distance = torch.sqrt(
            (pred_x.float() - target_x.float()) ** 2
            +
            (pred_y.float() - target_y.float()) ** 2
        )

        correct = (
            distance <= threshold
        )

        return correct.float().mean().item()


# =========================================================
# Save checkpoint
# =========================================================

def save_checkpoint(
    completed_epoch,
    train_loss,
    val_loss,
    val_pck
):
    checkpoint = {
        "epoch_number": completed_epoch,
        "epoch": completed_epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "train_loss":
            float(train_loss),

        "validation_loss":
            float(val_loss),

        "val_pck":
            float(val_pck),

        "best_pck":
            float(best_pck),

        "best_val_loss":
            float(best_val_loss)
    }

    # -----------------------------------------------------
    # Latest resumable checkpoint
    # -----------------------------------------------------

    torch.save(
        checkpoint,
        LAST_CHECKPOINT
    )

    # -----------------------------------------------------
    # Individual epoch checkpoint
    # -----------------------------------------------------

    epoch_path = os.path.join(
        CHECKPOINT_DIR,
        f"vit_pose_epoch_{completed_epoch}.pth"
    )

    torch.save(
        checkpoint,
        epoch_path
    )

    return epoch_path


# =========================================================
# AMP helper
# =========================================================

def autocast_context():
    if USE_AMP:
        try:
            return torch.amp.autocast(
                device_type="cuda",
                enabled=True
            )
        except AttributeError:
            return torch.cuda.amp.autocast(
                enabled=True
            )

    return torch.autocast(
        device_type="cpu",
        enabled=False
    )


# =========================================================
# Training
# =========================================================

try:

    for epoch in range(
        start_epoch,
        EPOCHS
    ):

        epoch_number = epoch + 1

        print("\n")
        print("=" * 70)
        print(
            f"Epoch {epoch_number}/{EPOCHS}"
        )
        print("=" * 70)

        epoch_start = time.time()

        # -------------------------------------------------
        # Training
        # -------------------------------------------------

        model.train()

        running_loss = 0.0
        running_pck = 0.0

        for batch_idx, batch in enumerate(
            train_loader
        ):

            # Dataset is expected to return:
            # image, heatmap
            images, targets = batch

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------

            with autocast_context():

                predictions = model(
                    images
                )

                loss = criterion(
                    predictions,
                    targets
                )

            # -------------------------------------------------
            # Backward
            # -------------------------------------------------

            if USE_AMP:

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------

            batch_loss = loss.item()

            batch_pck = heatmap_pck(
                predictions,
                targets
            )

            running_loss += batch_loss
            running_pck += batch_pck

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------

            if (
                batch_idx % 50 == 0
                or batch_idx == len(train_loader) - 1
            ):

                current_loss = (
                    running_loss /
                    (batch_idx + 1)
                )

                current_pck = (
                    running_pck /
                    (batch_idx + 1)
                ) * 100.0

                print(
                    f"Epoch [{epoch_number}/{EPOCHS}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {current_loss:.6f} "
                    f"PCK: {current_pck:.2f}%"
                )

        train_loss = (
            running_loss /
            max(1, len(train_loader))
        )

        train_pck = (
            running_pck /
            max(1, len(train_loader))
        ) * 100.0

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        model.eval()

        validation_loss = 0.0
        validation_pck = 0.0

        with torch.no_grad():

            for images, targets in val_loader:

                images = images.to(
                    DEVICE,
                    non_blocking=True
                )

                targets = targets.to(
                    DEVICE,
                    non_blocking=True
                )

                with autocast_context():

                    predictions = model(
                        images
                    )

                    loss = criterion(
                        predictions,
                        targets
                    )

                validation_loss += loss.item()

                validation_pck += heatmap_pck(
                    predictions,
                    targets
                )

        validation_loss /= max(
            1,
            len(val_loader)
        )

        validation_pck = (
            validation_pck /
            max(1, len(val_loader))
        ) * 100.0

        # -------------------------------------------------
        # Scheduler
        # -------------------------------------------------

        scheduler.step()

        # -------------------------------------------------
        # Best values
        # -------------------------------------------------

        new_best_pck = (
            validation_pck > best_pck
        )

        new_best_loss = (
            validation_loss < best_val_loss
        )

        if new_best_pck:
            best_pck = validation_pck

        if new_best_loss:
            best_val_loss = validation_loss

        # -------------------------------------------------
        # Save EVERY epoch
        # -------------------------------------------------

        epoch_path = save_checkpoint(
            completed_epoch=epoch_number,
            train_loss=train_loss,
            val_loss=validation_loss,
            val_pck=validation_pck
        )

        # -------------------------------------------------
        # Save best PCK model
        # -------------------------------------------------

        if new_best_pck:

            best_checkpoint = {
                "epoch_number":
                    epoch_number,

                "epoch":
                    epoch_number,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "scaler_state_dict":
                    scaler.state_dict(),

                "train_loss":
                    float(train_loss),

                "validation_loss":
                    float(validation_loss),

                "val_pck":
                    float(validation_pck),

                "best_pck":
                    float(best_pck),

                "best_val_loss":
                    float(best_val_loss)
            }

            torch.save(
                best_checkpoint,
                BEST_MODEL
            )

        # -------------------------------------------------
        # Save best loss model
        # -------------------------------------------------

        if new_best_loss:

            loss_checkpoint = {
                "epoch_number":
                    epoch_number,

                "epoch":
                    epoch_number,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "scaler_state_dict":
                    scaler.state_dict(),

                "train_loss":
                    float(train_loss),

                "validation_loss":
                    float(validation_loss),

                "val_pck":
                    float(validation_pck),

                "best_pck":
                    float(best_pck),

                "best_val_loss":
                    float(best_val_loss)
            }

            torch.save(
                loss_checkpoint,
                BEST_LOSS_MODEL
            )

        # -------------------------------------------------
        # Summary
        # -------------------------------------------------

        epoch_time = (
            time.time() -
            epoch_start
        )

        print("\n" + "-" * 70)
        print(
            f"Epoch {epoch_number} Completed"
        )
        print(
            f"Training Loss   : {train_loss:.6f}"
        )
        print(
            f"Training PCK    : {train_pck:.2f}%"
        )
        print(
            f"Validation Loss : {validation_loss:.6f}"
        )
        print(
            f"Validation PCK  : {validation_pck:.2f}%"
        )
        print(
            f"Best PCK        : {best_pck:.2f}%"
        )
        print(
            f"Best Val Loss   : {best_val_loss:.6f}"
        )
        print(
            f"Learning Rate   : "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )
        print(
            f"Epoch Time      : "
            f"{epoch_time / 60:.2f} min"
        )
        print(
            f"Saved Checkpoint: {epoch_path}"
        )
        print("-" * 70)

        # -------------------------------------------------
        # Reset CUDA peak memory counter
        # -------------------------------------------------

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

except KeyboardInterrupt:

    print("\n")
    print("=" * 70)
    print("TRAINING INTERRUPTED")
    print("=" * 70)

    # Save current state so the user can resume.
    interrupted_checkpoint = {
        "epoch_number":
            max(0, epoch),

        "epoch":
            max(0, epoch),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "best_pck":
            float(best_pck),

        "best_val_loss":
            float(best_val_loss)
    }

    torch.save(
        interrupted_checkpoint,
        LAST_CHECKPOINT
    )

    print(
        "Emergency checkpoint saved:"
    )
    print(
        LAST_CHECKPOINT
    )

    print(
        "\nRun the same command again to resume:"
    )
    print(
        "python -m training.train_pose"
    )

    raise

# =========================================================
# Finished
# =========================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best Validation PCK : {best_pck:.2f}%"
)

print(
    f"Best Validation Loss: {best_val_loss:.6f}"
)

print(
    "Best Model          :",
    BEST_MODEL
)

print(
    "Last Checkpoint     :",
    LAST_CHECKPOINT
)

print("=" * 70)
''')

path = "/mnt/data/train_pose.py"
Path(path).write_text(code, encoding="utf-8")
print(path)
