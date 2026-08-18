# ============================================================
# STAGE 4 - EXPERIMENT 2
# POSE SEQUENCE TRANSFORMER + VAE + ACTIVITY CLASSIFIER
# ============================================================

import os
import random
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

SEQUENCE_LENGTH = 16
NUM_JOINTS = 16
COORDINATES = 2

INPUT_DIM = NUM_JOINTS * COORDINATES     # 32

D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
FF_DIM = 256

LATENT_DIM = 64

DROPOUT = 0.20

BATCH_SIZE = 128
NUM_EPOCHS = 50

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# ------------------------------------------------------------
# VAE / classification loss
# ------------------------------------------------------------

# Classification receives strong priority.
CLASSIFICATION_WEIGHT = 1.0

# Reconstruction is useful but should not dominate.
RECONSTRUCTION_WEIGHT = 0.10

# Maximum KL contribution.
MAX_KL_WEIGHT = 0.001

# Number of epochs used to warm up KL.
KL_WARMUP_EPOCHS = 15

LABEL_SMOOTHING = 0.05

# ------------------------------------------------------------
# Checkpoint directory
# ------------------------------------------------------------

CHECKPOINT_DIR_NAME = "activity_exp2"

# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PROJECT ROOT
# ============================================================

def find_project_root():

    # Kaggle
    kaggle_root = "/kaggle/working/humanactivityrecognition"

    if os.path.exists(kaggle_root):
        return kaggle_root

    # Current file:
    # project/training/train_activity.py
    current_file = os.path.abspath(__file__)

    project_root = os.path.dirname(
        os.path.dirname(current_file)
    )

    return project_root


ROOT = find_project_root()

DATA_ROOT = os.path.join(
    ROOT,
    "datasets",
    "pose_sequence"
)

TRAIN_CSV = os.path.join(
    DATA_ROOT,
    "train.csv"
)

VAL_CSV = os.path.join(
    DATA_ROOT,
    "val.csv"
)

TEST_CSV = os.path.join(
    DATA_ROOT,
    "test.csv"
)

CLASSES_FILE = os.path.join(
    DATA_ROOT,
    "classes.txt"
)

CHECKPOINT_DIR = os.path.join(
    ROOT,
    "checkpoints",
    CHECKPOINT_DIR_NAME
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"

print("=" * 70)
print("STAGE 4 - EXPERIMENT 2")
print("POSE SEQUENCE TRANSFORMER + VAE")
print("=" * 70)

print("\nProject Root :", ROOT)
print("Device       :", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU          :",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA         :",
        torch.version.cuda
    )


# ============================================================
# VERIFY DATA
# ============================================================

required_files = [
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    CLASSES_FILE
]

print("\nChecking required files...")

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print("✓", path)


# ============================================================
# LOAD CLASSES
# ============================================================

with open(
    CLASSES_FILE,
    "r",
    encoding="utf-8"
) as f:

    classes = [
        line.strip()
        for line in f
        if line.strip()
    ]

NUM_CLASSES = len(classes)

print("\nNumber of classes :", NUM_CLASSES)


# ============================================================
# DATASET
# ============================================================

class PoseSequenceDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

        required_columns = [
            "path",
            "label",
            "class_name"
        ]

        for column in required_columns:

            if column not in self.df.columns:

                raise ValueError(
                    f"Column '{column}' missing "
                    f"from {csv_file}"
                )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        path = str(row["path"])

        # Handle both absolute and relative paths.
        if not os.path.isabs(path):

            path = os.path.join(
                ROOT,
                path
            )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Pose sequence not found:\n{path}"
            )

        sequence = np.load(
            path
        ).astype(np.float32)

        # ----------------------------------------------------
        # Expected:
        # (16, 16, 2)
        # ----------------------------------------------------

        if sequence.shape != (
            SEQUENCE_LENGTH,
            NUM_JOINTS,
            COORDINATES
        ):

            raise ValueError(
                f"Invalid sequence shape "
                f"{sequence.shape} in {path}"
            )

        # ----------------------------------------------------
        # Convert:
        #
        # (16, 16, 2)
        #
        # to:
        #
        # (16, 32)
        # ----------------------------------------------------

        sequence = sequence.reshape(
            SEQUENCE_LENGTH,
            INPUT_DIM
        )

        x = torch.from_numpy(
            sequence
        )

        y = int(
            row["label"]
        )

        return x, y


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset = PoseSequenceDataset(
    TRAIN_CSV
)

val_dataset = PoseSequenceDataset(
    VAL_CSV
)

test_dataset = PoseSequenceDataset(
    TEST_CSV
)

print(
    "Training sequences   :",
    len(train_dataset)
)

print(
    "Validation sequences :",
    len(val_dataset)
)

print(
    "Test sequences       :",
    len(test_dataset)
)


# ============================================================
# DATALOADERS
# ============================================================

PIN_MEMORY = DEVICE.type == "cuda"

NUM_WORKERS = 2 if DEVICE.type == "cuda" else 0

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=NUM_WORKERS > 0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=NUM_WORKERS > 0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=NUM_WORKERS > 0
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

train_labels = train_dataset.df["label"].astype(int)

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

class_counts = np.maximum(
    class_counts,
    1
)

# Balanced weighting.
weights = (
    len(train_labels)
    /
    (
        NUM_CLASSES *
        class_counts
    )
)

weights = torch.tensor(
    weights,
    dtype=torch.float32
)

weights = weights.to(DEVICE)

print("✓ Class weights calculated")


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=16
    ):

        super().__init__()

        position = torch.arange(
            max_len
        ).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            *
            (
                -np.log(10000.0)
                /
                d_model
            )
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        return x + self.pe[
            :, :x.size(1)
        ]


# ============================================================
# TRANSFORMER + VAE MODEL
# ============================================================

class PoseTransformerVAE(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        ff_dim,
        latent_dim,
        num_classes,
        dropout
    ):

        super().__init__()

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

        self.input_projection = nn.Sequential(

            nn.Linear(
                input_dim,
                d_model
            ),

            nn.LayerNorm(
                d_model
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            )
        )

        # ----------------------------------------------------
        # Positional encoding
        # ----------------------------------------------------

        self.position = PositionalEncoding(
            d_model,
            SEQUENCE_LENGTH
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
        )

        self.final_norm = nn.LayerNorm(
            d_model
        )

        # ----------------------------------------------------
        # VAE
        # ----------------------------------------------------

        self.fc_mu = nn.Linear(
            d_model,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            d_model,
            latent_dim
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
                d_model
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_model,
                SEQUENCE_LENGTH *
                input_dim
            )
        )

        # ----------------------------------------------------
        # Activity classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.LayerNorm(
                latent_dim
            ),

            nn.Linear(
                latent_dim,
                d_model
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_model,
                num_classes
            )
        )

    def encode(
        self,
        x
    ):

        x = self.input_projection(
            x
        )

        x = self.position(
            x
        )

        x = self.transformer(
            x
        )

        x = self.final_norm(
            x
        )

        # Temporal mean pooling
        x = x.mean(
            dim=1
        )

        mu = self.fc_mu(
            x
        )

        logvar = self.fc_logvar(
            x
        )

        return mu, logvar

    def reparameterize(
        self,
        mu,
        logvar
    ):

        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(
            std
        )

        return (
            mu +
            eps * std
        )

    def forward(
        self,
        x
    ):

        mu, logvar = self.encode(
            x
        )

        z = self.reparameterize(
            mu,
            logvar
        )

        reconstruction = self.decoder(
            z
        )

        reconstruction = reconstruction.reshape(
            -1,
            SEQUENCE_LENGTH,
            INPUT_DIM
        )

        logits = self.classifier(
            z
        )

        return (
            logits,
            reconstruction,
            mu,
            logvar
        )


# ============================================================
# MODEL
# ============================================================

print("\nBuilding model...")

model = PoseTransformerVAE(
    input_dim=INPUT_DIM,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    latent_dim=LATENT_DIM,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(DEVICE)

print("✓ Model loaded")


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
    eta_min=1e-6
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=weights,
    label_smoothing=LABEL_SMOOTHING
)


# ============================================================
# AMP
# ============================================================

scaler = GradScaler(
    enabled=USE_AMP
)


# ============================================================
# CHECKPOINT VARIABLES
# ============================================================

START_EPOCH = 1

BEST_VAL_ACC = 0.0
BEST_VAL_LOSS = float("inf")

LAST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "last_checkpoint.pth"
)

BEST_MODEL = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)


# ============================================================
# RESUME
# ============================================================

if os.path.exists(
    LAST_CHECKPOINT
):

    print("\n" + "=" * 70)
    print("RESUMING EXPERIMENT 2")
    print("=" * 70)

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

    if (
        "scheduler_state_dict"
        in checkpoint
    ):

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

    if (
        "scaler_state_dict"
        in checkpoint
    ):

        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )

    START_EPOCH = (
        checkpoint["epoch"] + 1
    )

    BEST_VAL_ACC = checkpoint.get(
        "best_val_acc",
        0.0
    )

    BEST_VAL_LOSS = checkpoint.get(
        "best_val_loss",
        float("inf")
    )

    print(
        "Last Epoch :",
        checkpoint["epoch"]
    )

    print(
        "Next Epoch :",
        START_EPOCH
    )

    print(
        "Best Val Accuracy :",
        f"{BEST_VAL_ACC:.2f}%"
    )

else:

    print("\nNo previous Experiment-2 checkpoint.")
    print("Starting from Epoch 1.")


# ============================================================
# KL WEIGHT
# ============================================================

def get_kl_weight(epoch):

    if KL_WARMUP_EPOCHS <= 0:

        return MAX_KL_WEIGHT

    progress = min(
        epoch / KL_WARMUP_EPOCHS,
        1.0
    )

    return (
        MAX_KL_WEIGHT *
        progress
    )


# ============================================================
# KL LOSS
# ============================================================

def kl_loss(
    mu,
    logvar
):

    value = -0.5 * torch.sum(
        1 +
        logvar -
        mu.pow(2) -
        logvar.exp(),
        dim=1
    )

    return value.mean()


# ============================================================
# TOP-K ACCURACY
# ============================================================

def topk_accuracy(
    logits,
    targets,
    k=1
):

    with torch.no_grad():

        max_k = min(
            k,
            logits.size(1)
        )

        _, pred = logits.topk(
            max_k,
            dim=1
        )

        correct = (
            pred ==
            targets.unsqueeze(1)
        )

        correct = correct.any(
            dim=1
        )

        return (
            correct.float().sum().item()
        )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    epoch
):

    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_recon = 0.0
    total_kl = 0.0

    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    kl_weight = get_kl_weight(
        epoch
    )

    start_time = time.time()

    for batch_idx, (
        x,
        y
    ) in enumerate(train_loader):

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            enabled=USE_AMP
        ):

            (
                logits,
                reconstruction,
                mu,
                logvar
            ) = model(x)

            ce = criterion(
                logits,
                y
            )

            recon = F.mse_loss(
                reconstruction,
                x
            )

            kl = kl_loss(
                mu,
                logvar
            )

            loss = (
                CLASSIFICATION_WEIGHT * ce
                +
                RECONSTRUCTION_WEIGHT * recon
                +
                kl_weight * kl
            )

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

        batch_size = x.size(0)

        total_samples += batch_size

        total_loss += (
            loss.item() *
            batch_size
        )

        total_ce += (
            ce.item() *
            batch_size
        )

        total_recon += (
            recon.item() *
            batch_size
        )

        total_kl += (
            kl.item() *
            batch_size
        )

        correct_top1 += topk_accuracy(
            logits,
            y,
            1
        )

        correct_top5 += topk_accuracy(
            logits,
            y,
            5
        )

        if (
            batch_idx % 50 == 0
            or
            batch_idx == len(train_loader) - 1
        ):

            running_acc = (
                100.0 *
                correct_top1 /
                total_samples
            )

            print(
                f"Epoch {epoch:03d} | "
                f"Batch {batch_idx:04d}/"
                f"{len(train_loader)-1:04d} | "
                f"Loss {loss.item():.4f} | "
                f"Top1 {running_acc:.2f}%"
            )

    return {
        "loss":
            total_loss /
            total_samples,

        "ce":
            total_ce /
            total_samples,

        "recon":
            total_recon /
            total_samples,

        "kl":
            total_kl /
            total_samples,

        "top1":
            100.0 *
            correct_top1 /
            total_samples,

        "top5":
            100.0 *
            correct_top5 /
            total_samples,

        "kl_weight":
            kl_weight,

        "time":
            time.time() -
            start_time
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    loader
):

    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_recon = 0.0
    total_kl = 0.0

    correct_top1 = 0
    correct_top5 = 0

    total_samples = 0

    for x, y in loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        with autocast(
            enabled=USE_AMP
        ):

            (
                logits,
                reconstruction,
                mu,
                logvar
            ) = model(x)

            ce = criterion(
                logits,
                y
            )

            recon = F.mse_loss(
                reconstruction,
                x
            )

            kl = kl_loss(
                mu,
                logvar
            )

            # During evaluation use
            # final KL weight.
            loss = (
                CLASSIFICATION_WEIGHT * ce
                +
                RECONSTRUCTION_WEIGHT * recon
                +
                MAX_KL_WEIGHT * kl
            )

        batch_size = x.size(0)

        total_samples += batch_size

        total_loss += (
            loss.item() *
            batch_size
        )

        total_ce += (
            ce.item() *
            batch_size
        )

        total_recon += (
            recon.item() *
            batch_size
        )

        total_kl += (
            kl.item() *
            batch_size
        )

        correct_top1 += topk_accuracy(
            logits,
            y,
            1
        )

        correct_top5 += topk_accuracy(
            logits,
            y,
            5
        )

    return {
        "loss":
            total_loss /
            total_samples,

        "ce":
            total_ce /
            total_samples,

        "recon":
            total_recon /
            total_samples,

        "kl":
            total_kl /
            total_samples,

        "top1":
            100.0 *
            correct_top1 /
            total_samples,

        "top5":
            100.0 *
            correct_top5 /
            total_samples
    }


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 70)
print("STARTING EXPERIMENT 2")
print("=" * 70)

print(
    "\nCheckpoint directory:",
    CHECKPOINT_DIR
)

for epoch in range(
    START_EPOCH,
    NUM_EPOCHS + 1
):

    epoch_start = time.time()

    print("\n")
    print("=" * 70)
    print(
        f"EPOCH {epoch}/{NUM_EPOCHS}"
    )
    print("=" * 70)

    train_metrics = train_one_epoch(
        epoch
    )

    val_metrics = evaluate(
        val_loader
    )

    scheduler.step()

    current_lr = optimizer.param_groups[
        0
    ]["lr"]

    print("\n" + "-" * 70)

    print(
        f"Epoch {epoch} Completed"
    )

    print(
        f"Training Loss       : "
        f"{train_metrics['loss']:.6f}"
    )

    print(
        f"Training Top-1      : "
        f"{train_metrics['top1']:.2f}%"
    )

    print(
        f"Training Top-5      : "
        f"{train_metrics['top5']:.2f}%"
    )

    print(
        f"Validation Loss     : "
        f"{val_metrics['loss']:.6f}"
    )

    print(
        f"Validation Top-1    : "
        f"{val_metrics['top1']:.2f}%"
    )

    print(
        f"Validation Top-5    : "
        f"{val_metrics['top5']:.2f}%"
    )

    print(
        f"CE Loss             : "
        f"{val_metrics['ce']:.6f}"
    )

    print(
        f"Reconstruction Loss : "
        f"{val_metrics['recon']:.6f}"
    )

    print(
        f"KL Loss             : "
        f"{val_metrics['kl']:.6f}"
    )

    print(
        f"KL Weight           : "
        f"{train_metrics['kl_weight']:.6f}"
    )

    print(
        f"Learning Rate       : "
        f"{current_lr:.8f}"
    )

    print(
        f"Time                : "
        f"{time.time()-epoch_start:.2f} sec"
    )

    # --------------------------------------------------------
    # Save every epoch
    # --------------------------------------------------------

    epoch_checkpoint = os.path.join(
        CHECKPOINT_DIR,
        f"epoch_{epoch:03d}.pth"
    )

    torch.save(
        {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "scaler_state_dict":
                scaler.state_dict(),

            "best_val_acc":
                BEST_VAL_ACC,

            "best_val_loss":
                BEST_VAL_LOSS,

            "train_metrics":
                train_metrics,

            "val_metrics":
                val_metrics,

            "classes":
                classes,

            "config":
                {
                    "sequence_length":
                        SEQUENCE_LENGTH,

                    "num_joints":
                        NUM_JOINTS,

                    "coordinates":
                        COORDINATES,

                    "d_model":
                        D_MODEL,

                    "nhead":
                        NHEAD,

                    "num_layers":
                        NUM_LAYERS,

                    "latent_dim":
                        LATENT_DIM
                }
        },
        epoch_checkpoint
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    improved = False

    if (
        val_metrics["top1"]
        >
        BEST_VAL_ACC
    ):

        BEST_VAL_ACC = (
            val_metrics["top1"]
        )

        improved = True

    if (
        val_metrics["loss"]
        <
        BEST_VAL_LOSS
    ):

        BEST_VAL_LOSS = (
            val_metrics["loss"]
        )

    if improved:

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "scaler_state_dict":
                    scaler.state_dict(),

                "best_val_acc":
                    BEST_VAL_ACC,

                "best_val_loss":
                    BEST_VAL_LOSS,

                "val_metrics":
                    val_metrics,

                "classes":
                    classes
            },
            BEST_MODEL
        )

        print(
            "\n✓ BEST MODEL UPDATED"
        )

        print(
            "Best Validation Top-1 :",
            f"{BEST_VAL_ACC:.2f}%"
        )

    # --------------------------------------------------------
    # Last checkpoint
    # --------------------------------------------------------

    torch.save(
        {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "scaler_state_dict":
                scaler.state_dict(),

            "best_val_acc":
                BEST_VAL_ACC,

            "best_val_loss":
                BEST_VAL_LOSS,

            "train_metrics":
                train_metrics,

            "val_metrics":
                val_metrics,

            "classes":
                classes
        },
        LAST_CHECKPOINT
    )

    print(
        "\nCheckpoint Saved:",
        epoch_checkpoint
    )


# ============================================================
# FINAL TEST
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"\nBest Validation Accuracy : "
    f"{BEST_VAL_ACC:.2f}%"
)

print(
    "Best Model :",
    BEST_MODEL
)


print("\n")
print("=" * 70)
print("LOADING BEST MODEL FOR TEST")
print("=" * 70)

if not os.path.exists(
    BEST_MODEL
):

    raise FileNotFoundError(
        "Best model was not created."
    )

best_checkpoint = torch.load(
    BEST_MODEL,
    map_location=DEVICE
)

model.load_state_dict(
    best_checkpoint[
        "model_state_dict"
    ]
)

print(
    "Best checkpoint epoch :",
    best_checkpoint.get(
        "epoch",
        "unknown"
    )
)

print(
    "Best validation accuracy :",
    f"{best_checkpoint.get('best_val_acc', 0.0):.2f}%"
)


test_metrics = evaluate(
    test_loader
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST RESULTS - EXPERIMENT 2")
print("=" * 70)

print(
    f"\nTest Loss       : "
    f"{test_metrics['loss']:.6f}"
)

print(
    f"Test Top-1      : "
    f"{test_metrics['top1']:.2f}%"
)

print(
    f"Test Top-5      : "
    f"{test_metrics['top5']:.2f}%"
)

print(
    f"Test CE Loss    : "
    f"{test_metrics['ce']:.6f}"
)

print(
    f"Test Recon Loss : "
    f"{test_metrics['recon']:.6f}"
)

print(
    f"Test KL Loss    : "
    f"{test_metrics['kl']:.6f}"
)


# ============================================================
# COMPARISON WITH EXPERIMENT 1
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT COMPARISON")
print("=" * 70)

print(
    "\nExperiment 1 Test Accuracy : 14.88%"
)

print(
    f"Experiment 2 Test Accuracy : "
    f"{test_metrics['top1']:.2f}%"
)

improvement = (
    test_metrics["top1"]
    - 14.88
)

print(
    f"Absolute Improvement       : "
    f"{improvement:+.2f} percentage points"
)

print("\n" + "=" * 70)
print("STAGE 4 EXPERIMENT 2 COMPLETED")
print("=" * 70)