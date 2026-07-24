"""
=========================================================
ViTPose Training Script
Supports:

✓ Resume Training
✓ Best Model Saving
✓ Last Checkpoint Saving
✓ Kaggle GPU & Windows CPU/GPU Compatibility
✓ Mixed Precision Training
✓ LR Scheduler
✓ Gradient Clipping

=========================================================
"""

import os
import sys
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms

# Updated imports for newer PyTorch versions
from torch.amp import autocast, GradScaler

# -------------------------------------------------------
# Project Root & Config
# -------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from configs.config import Config
cfg = Config

# =========================================================
# Main Execution Block (Required for Windows DataLoader)
# =========================================================

if __name__ == "__main__":

    # -------------------------------------------------------
    # Dataset & Transformations
    # -------------------------------------------------------
    from datasets.mpii_train_dataset import MPIITrainDataset

    # FIXED: ToTensor() must come FIRST to convert the NumPy array 
    # before applying Resize() and Normalize()
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), antialias=True),
        transforms.Normalize(mean=cfg.MEAN, std=cfg.STD)
    ])

    # Pass the transform pipeline into MPIITrainDataset
    train_dataset = MPIITrainDataset(
        image_dir=cfg.MPII_IMAGE_DIR,
        annotation_file=cfg.MPII_ANNOTATION_FILE,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY and torch.cuda.is_available()
    )

    # -------------------------------------------------------
    # Model
    # -------------------------------------------------------
    from models.vit_pose import ViTPose

    DEVICE = cfg.DEVICE

    model = ViTPose(
        num_joints=cfg.NUM_JOINTS
    ).to(DEVICE)

    # -------------------------------------------------------
    # Loss, Optimizer & Scheduler
    # -------------------------------------------------------
    criterion = nn.MSELoss()

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.POSE_EPOCHS
    )

    # -------------------------------------------------------
    # Mixed Precision (Automatically disables if on CPU)
    # -------------------------------------------------------
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())

    # -------------------------------------------------------
    # Checkpoint Directory
    # -------------------------------------------------------
    os.makedirs(
        cfg.CHECKPOINT_DIR,
        exist_ok=True
    )

    LAST_CHECKPOINT = os.path.join(
        cfg.CHECKPOINT_DIR,
        "last_checkpoint.pth"
    )

    BEST_MODEL = os.path.join(
        cfg.CHECKPOINT_DIR,
        "best_model.pth"
    )

    best_loss = float("inf")
    start_epoch = 0

    print("=" * 60)
    print("Device :", DEVICE)
    print("=" * 60)

    # =========================================================
    # Resume Training Automatically
    # =========================================================

    if os.path.exists(LAST_CHECKPOINT):

        print("=" * 60)
        print("Loading Previous Checkpoint...")
        print("=" * 60)

        checkpoint = torch.load(
            LAST_CHECKPOINT,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]

        print(f"Resuming from Epoch : {start_epoch}")
        print(f"Best Loss           : {best_loss:.6f}")

    else:

        print("=" * 60)
        print("No Previous Checkpoint Found")
        print("Training From Scratch")
        print("=" * 60)

    # =========================================================
    # Training Information
    # =========================================================

    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)

    print("Epochs        :", cfg.POSE_EPOCHS)
    print("Batch Size    :", cfg.BATCH_SIZE)
    print("Learning Rate :", cfg.LEARNING_RATE)
    print("Device        :", DEVICE)

    print("=" * 60)

    # =========================================================
    # Start Training
    # =========================================================

    try:
        for epoch in range(start_epoch, cfg.POSE_EPOCHS):

            model.train()
            running_loss = 0.0
            epoch_start = time.time()

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch [{epoch+1}/{cfg.POSE_EPOCHS}]",
                leave=True
            )

            for images, targets in progress_bar:
                images = images.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # Mixed Precision Forward Pass
                with autocast(device_type=DEVICE.type, enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                # Backward Pass
                scaler.scale(loss).backward()

                # Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer Step & Scaler Update
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            # End of Epoch Metrics
            epoch_loss = running_loss / len(train_loader)
            epoch_time = time.time() - epoch_start
            scheduler.step()

            print(f"\nEpoch [{epoch+1}/{cfg.POSE_EPOCHS}] Completed in {epoch_time:.2f}s | Average Loss: {epoch_loss:.6f}")

            # Save Last Checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_loss": best_loss,
            }, LAST_CHECKPOINT)

            # Save Best Model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), BEST_MODEL)
                print(f"--> Saved New Best Model! (Loss: {best_loss:.6f})")

            print("-" * 60)

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Training Interrupted by User (Ctrl+C)")
        print("Saving current checkpoint before exiting...")
        print("=" * 60)
        
        torch.save({
            "epoch": epoch if 'epoch' in locals() else start_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_loss": best_loss,
        }, LAST_CHECKPOINT)
        
        print(f"Checkpoint saved successfully to: {LAST_CHECKPOINT}")
        sys.exit(0)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        raise e