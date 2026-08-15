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

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
PCK_THRESHOLD = 5.0

TRAIN_SPLIT = 0.80

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = DEVICE.type == "cuda"

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
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=(NUM_WORKERS > 0)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=(NUM_WORKERS > 0)
)

# ---------------------------------------------------
# Model
# ---------------------------------------------------

model = ViTPose(
    num_joints=16
).to(DEVICE)

print("\nModel Loaded Successfully\n")
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

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
PCK_THRESHOLD = 5.0

TRAIN_SPLIT = 0.80

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = DEVICE.type == "cuda"

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
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=(NUM_WORKERS > 0)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=(NUM_WORKERS > 0)
)

# ---------------------------------------------------
# Model
# ---------------------------------------------------

model = ViTPose(
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
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

best_loss = float("inf")
start_epoch = 0
# ---------------------------------------------------
# Mixed Precision
# ---------------------------------------------------

if USE_AMP:
    scaler = torch.amp.GradScaler("cuda")
else:
    scaler = None

# =====================================================
# Automatic Resume
# =====================================================

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

best_loss = float("inf")
best_pck = 0.0
start_epoch = 0

if os.path.exists(LAST_CHECKPOINT):

    print("=" * 60)
    print("Loading last checkpoint...")
    print(LAST_CHECKPOINT)
    print("=" * 60)

    checkpoint = torch.load(
        LAST_CHECKPOINT,
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

    start_epoch = int(
        checkpoint.get(
            "epoch_number",
            checkpoint.get("epoch", 0)
        )
    )

    best_loss = float(
        checkpoint.get(
            "best_loss",
            checkpoint.get("best_val_loss", float("inf"))
        )
    )

    best_pck = float(
        checkpoint.get("best_pck", 0.0)
    )

    print("Completed Epoch :", start_epoch)
    print("Next Epoch      :", start_epoch + 1)
    print("Best Loss       :", best_loss)
    print("Best PCK        :", f"{best_pck:.2f}%")

else:

    print("No last checkpoint found.")
    print("Starting from Epoch 1.")


def heatmap_pck(predictions, targets, threshold=PCK_THRESHOLD):

    with torch.no_grad():

        b, j, h, w = predictions.shape

        pred_index = predictions.reshape(
            b, j, -1
        ).argmax(dim=-1)

        target_index = targets.reshape(
            b, j, -1
        ).argmax(dim=-1)

        pred_x = pred_index % w
        pred_y = pred_index // w

        target_x = target_index % w
        target_y = target_index // w

        distance = torch.sqrt(
            (pred_x.float() - target_x.float()) ** 2
            +
            (pred_y.float() - target_y.float()) ** 2
        )

        return (
            distance <= threshold
        ).float().mean().item()


def save_checkpoint(
    epoch_number,
    train_loss,
    validation_loss,
    validation_pck
):

    state = {
        "epoch": epoch_number,
        "epoch_number": epoch_number,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": float(train_loss),
        "validation_loss": float(validation_loss),
        "val_pck": float(validation_pck),
        "best_loss": float(best_loss),
        "best_pck": float(best_pck)
    }

    # Always overwrite this one so training can resume.
    torch.save(
        state,
        LAST_CHECKPOINT
    )

    # Save a full epoch checkpoint only every 10 epochs.
    # This prevents ~1 GB of storage being consumed per epoch.
    epoch_path = None

    if epoch_number % 10 == 0:
        epoch_path = os.path.join(
            CHECKPOINT_DIR,
            f"vit_pose_epoch_{epoch_number}.pth"
        )

        torch.save(
            state,
            epoch_path
        )

        print(f"Milestone checkpoint saved: {epoch_path}")

    return state, epoch_path


# =====================================================
# Training Loop
# =====================================================

try:

    for epoch in range(start_epoch, EPOCHS):

        print("\n")
        print("=" * 70)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("=" * 70)

        epoch_start = time.time()

        model.train()

        running_loss = 0.0
        running_pck = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):

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

            if USE_AMP:

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    predictions = model(images)

                    loss = criterion(
                        predictions,
                        targets
                    )

                if "scaler" not in globals():
                    scaler = torch.amp.GradScaler("cuda")

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                scaler.step(optimizer)
                scaler.update()

            else:

                predictions = model(images)

                loss = criterion(
                    predictions,
                    targets
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

            running_loss += loss.item()

            running_pck += heatmap_pck(
                predictions,
                targets
            )

            if batch_idx % 100 == 0:

                print(
                    f"Batch {batch_idx:5d}/{len(train_loader)}"
                    f" | Loss: {loss.item():.6f}"
                )

        train_loss = (
            running_loss /
            max(1, len(train_loader))
        )

        train_pck = (
            100.0 *
            running_pck /
            max(1, len(train_loader))
        )

        scheduler.step()

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

                if USE_AMP:

                    with torch.amp.autocast(
                        device_type="cuda"
                    ):

                        predictions = model(images)

                        loss = criterion(
                            predictions,
                            targets
                        )

                else:

                    predictions = model(images)

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
            100.0 *
            validation_pck /
            max(1, len(val_loader))
        )

        is_best_loss = (
            validation_loss < best_loss
        )

        is_best_pck = (
            validation_pck > best_pck
        )

        if is_best_loss:
            best_loss = validation_loss

        if is_best_pck:
            best_pck = validation_pck

        # -------------------------------------------------
        # Save every epoch
        # -------------------------------------------------

        state, checkpoint_path = save_checkpoint(
            epoch + 1,
            train_loss,
            validation_loss,
            validation_pck
        )

        # -------------------------------------------------
        # Save best models
        # -------------------------------------------------

        if is_best_loss:

            torch.save(
                state,
                BEST_LOSS_MODEL
            )

            print("Best loss model updated!")

        if is_best_pck:

            torch.save(
                state,
                BEST_MODEL
            )

            print("Best PCK model updated!")

        epoch_time = (
            time.time() - epoch_start
        )

        print("\n--------------------------------------")
        print(f"Epoch {epoch + 1} Completed")
        print(f"Training Loss   : {train_loss:.6f}")
        print(f"Training PCK    : {train_pck:.2f}%")
        print(f"Validation Loss : {validation_loss:.6f}")
        print(f"Validation PCK  : {validation_pck:.2f}%")
        print(f"Best Loss       : {best_loss:.6f}")
        print(f"Best PCK        : {best_pck:.2f}%")
        print(
            f"Learning Rate   : "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )
        print(f"Time            : {epoch_time / 60:.2f} min")
        if checkpoint_path is not None:
            print(f"Checkpoint Saved: {checkpoint_path}")
        else:
            print("Checkpoint Saved: last_checkpoint.pth (resume only)")
        print("--------------------------------------")

except KeyboardInterrupt:

    print("\nTraining interrupted.")
    print("The last completed epoch is saved.")
    print("Run the same command again to resume:")
    print("python -m training.train_pose")
    raise


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
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

best_loss = float("inf")
start_epoch = 0
# ---------------------------------------------------
# Mixed Precision
# ---------------------------------------------------

if USE_AMP:
    scaler = torch.amp.GradScaler("cuda")
else:
    scaler = None

# =====================================================
# Automatic Resume
# =====================================================

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

best_loss = float("inf")
best_pck = 0.0
start_epoch = 0

if os.path.exists(LAST_CHECKPOINT):

    print("=" * 60)
    print("Loading last checkpoint...")
    print(LAST_CHECKPOINT)
    print("=" * 60)

    checkpoint = torch.load(
        LAST_CHECKPOINT,
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

    start_epoch = int(
        checkpoint.get(
            "epoch_number",
            checkpoint.get("epoch", 0)
        )
    )

    best_loss = float(
        checkpoint.get(
            "best_loss",
            checkpoint.get("best_val_loss", float("inf"))
        )
    )

    best_pck = float(
        checkpoint.get("best_pck", 0.0)
    )

    print("Completed Epoch :", start_epoch)
    print("Next Epoch      :", start_epoch + 1)
    print("Best Loss       :", best_loss)
    print("Best PCK        :", f"{best_pck:.2f}%")

else:

    print("No last checkpoint found.")
    print("Starting from Epoch 1.")


def heatmap_pck(predictions, targets, threshold=PCK_THRESHOLD):

    with torch.no_grad():

        b, j, h, w = predictions.shape

        pred_index = predictions.reshape(
            b, j, -1
        ).argmax(dim=-1)

        target_index = targets.reshape(
            b, j, -1
        ).argmax(dim=-1)

        pred_x = pred_index % w
        pred_y = pred_index // w

        target_x = target_index % w
        target_y = target_index // w

        distance = torch.sqrt(
            (pred_x.float() - target_x.float()) ** 2
            +
            (pred_y.float() - target_y.float()) ** 2
        )

        return (
            distance <= threshold
        ).float().mean().item()


def save_checkpoint(
    epoch_number,
    train_loss,
    validation_loss,
    validation_pck
):

    state = {
        "epoch": epoch_number,
        "epoch_number": epoch_number,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": float(train_loss),
        "validation_loss": float(validation_loss),
        "val_pck": float(validation_pck),
        "best_loss": float(best_loss),
        "best_pck": float(best_pck)
    }

    # Always overwrite this one so training can resume.
    torch.save(
        state,
        LAST_CHECKPOINT
    )

    # Keep a separate file for EVERY epoch.
    epoch_path = os.path.join(
        CHECKPOINT_DIR,
        f"vit_pose_epoch_{epoch_number}.pth"
    )

    torch.save(
        state,
        epoch_path
    )

    return state, epoch_path


# =====================================================
# Training Loop
# =====================================================

try:

    for epoch in range(start_epoch, EPOCHS):

        print("\n")
        print("=" * 70)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("=" * 70)

        epoch_start = time.time()

        model.train()

        running_loss = 0.0
        running_pck = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):

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

            if USE_AMP:

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    predictions = model(images)

                    loss = criterion(
                        predictions,
                        targets
                    )

                if "scaler" not in globals():
                    scaler = torch.amp.GradScaler("cuda")

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                scaler.step(optimizer)
                scaler.update()

            else:

                predictions = model(images)

                loss = criterion(
                    predictions,
                    targets
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

            running_loss += loss.item()

            running_pck += heatmap_pck(
                predictions,
                targets
            )

            if batch_idx % 100 == 0:

                print(
                    f"Batch {batch_idx:5d}/{len(train_loader)}"
                    f" | Loss: {loss.item():.6f}"
                )

        train_loss = (
            running_loss /
            max(1, len(train_loader))
        )

        train_pck = (
            100.0 *
            running_pck /
            max(1, len(train_loader))
        )

        scheduler.step()

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

                if USE_AMP:

                    with torch.amp.autocast(
                        device_type="cuda"
                    ):

                        predictions = model(images)

                        loss = criterion(
                            predictions,
                            targets
                        )

                else:

                    predictions = model(images)

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
            100.0 *
            validation_pck /
            max(1, len(val_loader))
        )

        is_best_loss = (
            validation_loss < best_loss
        )

        is_best_pck = (
            validation_pck > best_pck
        )

        if is_best_loss:
            best_loss = validation_loss

        if is_best_pck:
            best_pck = validation_pck

        # -------------------------------------------------
        # Save every epoch
        # -------------------------------------------------

        state, checkpoint_path = save_checkpoint(
            epoch + 1,
            train_loss,
            validation_loss,
            validation_pck
        )

        # -------------------------------------------------
        # Save best models
        # -------------------------------------------------

        if is_best_loss:

            torch.save(
                state,
                BEST_LOSS_MODEL
            )

            print("Best loss model updated!")

        if is_best_pck:

            torch.save(
                state,
                BEST_MODEL
            )

            print("Best PCK model updated!")

        epoch_time = (
            time.time() - epoch_start
        )

        print("\n--------------------------------------")
        print(f"Epoch {epoch + 1} Completed")
        print(f"Training Loss   : {train_loss:.6f}")
        print(f"Training PCK    : {train_pck:.2f}%")
        print(f"Validation Loss : {validation_loss:.6f}")
        print(f"Validation PCK  : {validation_pck:.2f}%")
        print(f"Best Loss       : {best_loss:.6f}")
        print(f"Best PCK        : {best_pck:.2f}%")
        print(
            f"Learning Rate   : "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )
        print(f"Time            : {epoch_time / 60:.2f} min")
        print(f"Checkpoint Saved: {checkpoint_path}")
        print("--------------------------------------")

except KeyboardInterrupt:

    print("\nTraining interrupted.")
    print("The last completed epoch is saved.")
    print("Run the same command again to resume:")
    print("python -m training.train_pose")
    raise


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