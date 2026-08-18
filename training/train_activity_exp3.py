# ============================================================
# STAGE 4 - EXPERIMENT 3
# Temporal Transformer + Normalized Pose + Controlled VAE
# ============================================================

import os
import random
import time
import numpy as np
import pandas as pd

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

SEQ_LEN = 16
NUM_JOINTS = 16
COORDS = 2
NUM_CLASSES = 101

BATCH_SIZE = 128
EPOCHS = 50

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
FF_DIM = 256
DROPOUT = 0.20

# Controlled VAE
BETA_KL = 0.0001
RECON_WEIGHT = 0.05

NUM_WORKERS = 2

# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = ROOT / "datasets" / "pose_sequence"

TRAIN_CSV = DATA_ROOT / "train.csv"
VAL_CSV = DATA_ROOT / "val.csv"
TEST_CSV = DATA_ROOT / "test.csv"

CLASSES_FILE = DATA_ROOT / "classes.txt"

CHECKPOINT_DIR = ROOT / "checkpoints" / "activity_exp3"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL = CHECKPOINT_DIR / "best_model.pth"

LAST_MODEL = CHECKPOINT_DIR / "last_checkpoint.pth"

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("STAGE 4 - EXPERIMENT 3")
print("TEMPORAL TRANSFORMER + NORMALIZED POSE")
print("=" * 70)

print()
print("Project Root :", ROOT)
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
# CHECK FILES
# ============================================================

required_files = [
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    CLASSES_FILE
]

print()
print("Checking dataset files...")

for file in required_files:

    if not file.exists():

        raise FileNotFoundError(
            f"Required file not found: {file}"
        )

    print("✓", file)

# ============================================================
# LOAD CLASSES
# ============================================================

with open(CLASSES_FILE, "r", encoding="utf-8") as f:

    classes = [
        line.strip()
        for line in f
        if line.strip()
    ]

NUM_CLASSES = len(classes)

print()
print("Number of classes :", NUM_CLASSES)

# ============================================================
# DATASET
# ============================================================

class PoseSequenceDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

        print(
            f"Loaded {len(self.df)} samples from "
            f"{csv_file.name}"
        )

    def __len__(self):

        return len(self.df)

    def normalize_pose(self, pose):

        # ----------------------------------------------------
        # pose shape:
        # (16,16,2)
        # ----------------------------------------------------

        pose = pose.astype(
            np.float32
        )

        # ----------------------------------------------------
        # STEP 1: Root centering
        #
        # Use joint 0 as root.
        # ----------------------------------------------------

        root = pose[:, 0:1, :]

        pose = pose - root

        # ----------------------------------------------------
        # STEP 2: Scale normalization
        #
        # Calculate maximum joint distance.
        # ----------------------------------------------------

        distances = np.linalg.norm(
            pose,
            axis=-1
        )

        scale = np.max(
            distances
        )

        if scale > 1e-6:

            pose = pose / scale

        # ----------------------------------------------------
        # STEP 3: Clip abnormal values
        # ----------------------------------------------------

        pose = np.clip(
            pose,
            -3.0,
            3.0
        )

        return pose

    def __getitem__(self, index):

        row = self.df.iloc[index]

        path = row["path"]

        label = int(
            row["label"]
        )

        # ----------------------------------------------------
        # Handle relative paths
        # ----------------------------------------------------

        path = Path(path)

        if not path.is_absolute():

            path = ROOT / path

        if not path.exists():

            # Some CSVs may contain paths beginning with
            # outputs/...
            alternative = ROOT / path.name

            if alternative.exists():

                path = alternative

            else:

                raise FileNotFoundError(
                    f"Pose sequence not found: {path}"
                )

        pose = np.load(
            path
        )

        if pose.shape != (
            SEQ_LEN,
            NUM_JOINTS,
            COORDS
        ):

            raise ValueError(
                f"Invalid pose shape {pose.shape} "
                f"for {path}"
            )

        pose = self.normalize_pose(
            pose
        )

        pose = torch.tensor(
            pose,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return pose, label


# ============================================================
# LOAD DATASETS
# ============================================================

print()
print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

train_dataset = PoseSequenceDataset(
    TRAIN_CSV
)

val_dataset = PoseSequenceDataset(
    VAL_CSV
)

test_dataset = PoseSequenceDataset(
    TEST_CSV
)

# ============================================================
# CLASS WEIGHTS
# ============================================================

train_labels = train_dataset.df[
    "label"
].astype(int).values

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

class_weights = (
    len(train_labels)
    /
    (
        NUM_CLASSES *
        np.maximum(class_counts, 1)
    )
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)

print()
print("Class weighting enabled.")

# ============================================================
# WEIGHTED SAMPLER
# ============================================================

sample_weights = class_weights[
    torch.tensor(
        train_labels,
        dtype=torch.long
    )
]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0
)

print()
print("Train samples :", len(train_dataset))
print("Val samples   :", len(val_dataset))
print("Test samples  :", len(test_dataset))

# ============================================================
# MODEL
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=SEQ_LEN
    ):

        super().__init__()

        position = torch.arange(
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            )
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
# VAE BLOCK
# ============================================================

class PoseVAE(nn.Module):

    def __init__(self, input_dim, latent_dim=64):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.GELU(),

            nn.Linear(
                256,
                128
            ),

            nn.GELU()
        )

        self.mu = nn.Linear(
            128,
            latent_dim
        )

        self.logvar = nn.Linear(
            128,
            latent_dim
        )

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
                128
            ),

            nn.GELU(),

            nn.Linear(
                128,
                input_dim
            )
        )

    def forward(self, x):

        h = self.encoder(x)

        mu = self.mu(h)

        logvar = self.logvar(h)

        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(
            std
        )

        z = mu + eps * std

        reconstruction = self.decoder(
            z
        )

        return (
            reconstruction,
            mu,
            logvar
        )


# ============================================================
# TEMPORAL TRANSFORMER
# ============================================================

class ActivityTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        pose_dim = (
            NUM_JOINTS *
            COORDS
        )

        # ----------------------------------------------------
        # Pose projection
        # ----------------------------------------------------

        self.input_projection = nn.Sequential(

            nn.Linear(
                pose_dim,
                D_MODEL
            ),

            nn.LayerNorm(
                D_MODEL
            ),

            nn.GELU()
        )

        # ----------------------------------------------------
        # Positional encoding
        # ----------------------------------------------------

        self.position = PositionalEncoding(
            D_MODEL
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=D_MODEL,

            nhead=NHEAD,

            dim_feedforward=FF_DIM,

            dropout=DROPOUT,

            activation="gelu",

            batch_first=True,

            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=NUM_LAYERS
        )

        # ----------------------------------------------------
        # VAE
        # ----------------------------------------------------

        self.vae = PoseVAE(
            pose_dim,
            latent_dim=64
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.LayerNorm(
                D_MODEL
            ),

            nn.Linear(
                D_MODEL,
                256
            ),

            nn.GELU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                256,
                NUM_CLASSES
            )
        )

    def forward(self, pose):

        batch_size = pose.size(0)

        # ----------------------------------------------------
        # Flatten joints
        # ----------------------------------------------------

        x = pose.reshape(
            batch_size,
            SEQ_LEN,
            NUM_JOINTS * COORDS
        )

        # ----------------------------------------------------
        # VAE
        # ----------------------------------------------------

        vae_input = x.reshape(
            batch_size * SEQ_LEN,
            -1
        )

        reconstruction, mu, logvar = self.vae(
            vae_input
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        x = self.input_projection(
            x
        )

        x = self.position(
            x
        )

        x = self.transformer(
            x
        )

        # ----------------------------------------------------
        # Temporal average pooling
        # ----------------------------------------------------

        x = x.mean(
            dim=1
        )

        logits = self.classifier(
            x
        )

        return (
            logits,
            reconstruction,
            mu,
            logvar,
            vae_input
        )


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING TEMPORAL TRANSFORMER")
print("=" * 70)

model = ActivityTransformer().to(
    DEVICE
)

print(model)

# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(DEVICE),
    label_smoothing=0.1
)

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

    T_max=EPOCHS,

    eta_min=1e-6
)

# ============================================================
# LOSS FUNCTION
# ============================================================

def calculate_loss(
    logits,
    reconstruction,
    mu,
    logvar,
    original,
    labels
):

    ce_loss = criterion(
        logits,
        labels
    )

    recon_loss = F.mse_loss(
        reconstruction,
        original
    )

    # Stable KL calculation
    logvar = torch.clamp(
        logvar,
        min=-10.0,
        max=10.0
    )

    kl_loss = -0.5 * torch.mean(

        1
        +
        logvar
        -
        mu.pow(2)
        -
        logvar.exp()
    )

    total_loss = (

        ce_loss

        +

        RECON_WEIGHT *
        recon_loss

        +

        BETA_KL *
        kl_loss
    )

    return (
        total_loss,
        ce_loss,
        recon_loss,
        kl_loss
    )


# ============================================================
# ACCURACY
# ============================================================

def topk_accuracy(
    logits,
    labels,
    k=5
):

    _, predictions = torch.topk(
        logits,
        k,
        dim=1
    )

    correct = (
        predictions
        ==
        labels.unsqueeze(1)
    )

    return correct.any(
        dim=1
    ).float().mean().item()


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_recon = 0.0
    total_kl = 0.0

    correct = 0
    total = 0

    start = time.time()

    for batch_idx, (
        poses,
        labels
    ) in enumerate(train_loader):

        poses = poses.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            logits,
            reconstruction,
            mu,
            logvar,
            original
        ) = model(
            poses
        )

        (
            loss,
            ce_loss,
            recon_loss,
            kl_loss
        ) = calculate_loss(

            logits,
            reconstruction,
            mu,
            logvar,
            original,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item() *
            batch_size
        )

        total_ce += (
            ce_loss.item() *
            batch_size
        )

        total_recon += (
            recon_loss.item() *
            batch_size
        )

        total_kl += (
            kl_loss.item() *
            batch_size
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

        if batch_idx % 50 == 0:

            print(
                f"Batch {batch_idx:4d}/"
                f"{len(train_loader):4d} | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: "
                f"{100*correct/total:.2f}%"
            )

    elapsed = time.time() - start

    return {

        "loss":
        total_loss / total,

        "ce":
        total_ce / total,

        "recon":
        total_recon / total,

        "kl":
        total_kl / total,

        "accuracy":
        100 * correct / total,

        "time":
        elapsed
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()

def evaluate(loader):

    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_recon = 0.0
    total_kl = 0.0

    total = 0

    predictions_all = []
    labels_all = []

    top5_correct = 0

    for poses, labels in loader:

        poses = poses.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        (
            logits,
            reconstruction,
            mu,
            logvar,
            original
        ) = model(
            poses
        )

        (
            loss,
            ce_loss,
            recon_loss,
            kl_loss
        ) = calculate_loss(

            logits,
            reconstruction,
            mu,
            logvar,
            original,
            labels
        )

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            *
            batch_size
        )

        total_ce += (
            ce_loss.item()
            *
            batch_size
        )

        total_recon += (
            recon_loss.item()
            *
            batch_size
        )

        total_kl += (
            kl_loss.item()
            *
            batch_size
        )

        total += batch_size

        predictions = logits.argmax(
            dim=1
        )

        predictions_all.extend(
            predictions.cpu().numpy()
        )

        labels_all.extend(
            labels.cpu().numpy()
        )

        top5_correct += (

            torch.topk(
                logits,
                5,
                dim=1
            )
            .indices
            ==
            labels.unsqueeze(1)
        ).any(
            dim=1
        ).sum().item()

    accuracy = accuracy_score(
        labels_all,
        predictions_all
    )

    top5 = (
        top5_correct /
        total
    )

    macro_f1 = f1_score(
        labels_all,
        predictions_all,
        average="macro",
        zero_division=0
    )

    return {

        "loss":
        total_loss / total,

        "ce":
        total_ce / total,

        "recon":
        total_recon / total,

        "kl":
        total_kl / total,

        "accuracy":
        100 * accuracy,

        "top5":
        100 * top5,

        "f1":
        100 * macro_f1,

        "predictions":
        predictions_all,

        "labels":
        labels_all
    }


# ============================================================
# RESUME
# ============================================================

start_epoch = 1

best_accuracy = 0.0

best_f1 = 0.0

if LAST_MODEL.exists():

    print()
    print("=" * 70)
    print("RESUMING EXPERIMENT 3")
    print("=" * 70)

    checkpoint = torch.load(
        LAST_MODEL,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state"]
    )

    start_epoch = (
        checkpoint["epoch"] + 1
    )

    best_accuracy = checkpoint.get(
        "best_accuracy",
        0.0
    )

    best_f1 = checkpoint.get(
        "best_f1",
        0.0
    )

    print(
        "Resuming from epoch :",
        start_epoch
    )

    print(
        "Best validation accuracy :",
        best_accuracy
    )

else:

    print()
    print(
        "No previous Experiment 3 checkpoint found."
    )

# ============================================================
# TRAINING
# ============================================================

history = []

print()
print("=" * 70)
print("STARTING EXPERIMENT 3 TRAINING")
print("=" * 70)

for epoch in range(
    start_epoch,
    EPOCHS + 1
):

    print()
    print(
        "=" * 70
    )

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print(
        "=" * 70
    )

    train_metrics = train_one_epoch()

    val_metrics = evaluate(
        val_loader
    )

    scheduler.step()

    lr = optimizer.param_groups[
        0
    ]["lr"]

    print()
    print(
        f"Epoch {epoch} Completed"
    )

    print(
        f"Training Loss   : "
        f"{train_metrics['loss']:.6f}"
    )

    print(
        f"Training Acc    : "
        f"{train_metrics['accuracy']:.2f}%"
    )

    print(
        f"Validation Loss : "
        f"{val_metrics['loss']:.6f}"
    )

    print(
        f"Validation Top-1 : "
        f"{val_metrics['accuracy']:.2f}%"
    )

    print(
        f"Validation Top-5 : "
        f"{val_metrics['top5']:.2f}%"
    )

    print(
        f"Validation F1    : "
        f"{val_metrics['f1']:.2f}%"
    )

    print(
        f"CE Loss          : "
        f"{val_metrics['ce']:.6f}"
    )

    print(
        f"Recon Loss       : "
        f"{val_metrics['recon']:.6f}"
    )

    print(
        f"KL Loss          : "
        f"{val_metrics['kl']:.6f}"
    )

    print(
        f"Learning Rate     : "
        f"{lr:.8f}"
    )

    # --------------------------------------------------------
    # Save epoch checkpoint
    # --------------------------------------------------------

    epoch_checkpoint = (
        CHECKPOINT_DIR
        /
        f"epoch_{epoch:03d}.pth"
    )

    torch.save({

        "epoch":
        epoch,

        "model_state":
        model.state_dict(),

        "optimizer_state":
        optimizer.state_dict(),

        "scheduler_state":
        scheduler.state_dict(),

        "best_accuracy":
        best_accuracy,

        "best_f1":
        best_f1,

        "classes":
        classes

    }, epoch_checkpoint)

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_metrics["accuracy"] > best_accuracy:

        best_accuracy = (
            val_metrics["accuracy"]
        )

        best_f1 = (
            val_metrics["f1"]
        )

        torch.save({

            "epoch":
            epoch,

            "model_state":
            model.state_dict(),

            "optimizer_state":
            optimizer.state_dict(),

            "scheduler_state":
            scheduler.state_dict(),

            "best_accuracy":
            best_accuracy,

            "best_f1":
            best_f1,

            "classes":
            classes

        }, BEST_MODEL)

        print()
        print(
            "✓ BEST MODEL UPDATED"
        )

        print(
            f"Best Validation Accuracy : "
            f"{best_accuracy:.2f}%"
        )

    # --------------------------------------------------------
    # Save last checkpoint
    # --------------------------------------------------------

    torch.save({

        "epoch":
        epoch,

        "model_state":
        model.state_dict(),

        "optimizer_state":
        optimizer.state_dict(),

        "scheduler_state":
        scheduler.state_dict(),

        "best_accuracy":
        best_accuracy,

        "best_f1":
        best_f1,

        "classes":
        classes

    }, LAST_MODEL)

    history.append({

        "epoch":
        epoch,

        "train_loss":
        train_metrics["loss"],

        "train_accuracy":
        train_metrics["accuracy"],

        "val_loss":
        val_metrics["loss"],

        "val_accuracy":
        val_metrics["accuracy"],

        "val_top5":
        val_metrics["top5"],

        "val_f1":
        val_metrics["f1"],

        "ce_loss":
        val_metrics["ce"],

        "recon_loss":
        val_metrics["recon"],

        "kl_loss":
        val_metrics["kl"],

        "learning_rate":
        lr

    })

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_file = (
        CHECKPOINT_DIR
        /
        "training_history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_file,
        index=False
    )

    print(
        f"Checkpoint Saved: "
        f"{epoch_checkpoint}"
    )

# ============================================================
# TRAINING COMPLETED
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT 3 TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best Validation Accuracy : "
    f"{best_accuracy:.2f}%"
)

print(
    f"Best Validation Macro F1 : "
    f"{best_f1:.2f}%"
)

print(
    "Best Model :",
    BEST_MODEL
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL FOR TEST")
print("=" * 70)

checkpoint = torch.load(
    BEST_MODEL,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state"]
)

print(
    "Best checkpoint epoch :",
    checkpoint["epoch"]
)

print(
    f"Best validation accuracy : "
    f"{checkpoint['best_accuracy']:.2f}%"
)

# ============================================================
# FINAL TEST
# ============================================================

test_metrics = evaluate(
    test_loader
)

print()
print("=" * 70)
print("FINAL TEST RESULTS - EXPERIMENT 3")
print("=" * 70)

print(
    f"Test Loss       : "
    f"{test_metrics['loss']:.6f}"
)

print(
    f"Test Top-1      : "
    f"{test_metrics['accuracy']:.2f}%"
)

print(
    f"Test Top-5      : "
    f"{test_metrics['top5']:.2f}%"
)

print(
    f"Test Macro-F1   : "
    f"{test_metrics['f1']:.2f}%"
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
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(

    test_metrics["labels"],

    test_metrics["predictions"],

    target_names=classes,

    zero_division=0
)

print(report)

report_file = (
    CHECKPOINT_DIR
    /
    "classification_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    test_metrics["labels"],

    test_metrics["predictions"],

    labels=list(
        range(NUM_CLASSES)
    )
)

cm_file = (
    CHECKPOINT_DIR
    /
    "confusion_matrix.npy"
)

np.save(
    cm_file,
    cm
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT 3 COMPLETED")
print("=" * 70)

print(
    f"Best Validation Accuracy : "
    f"{best_accuracy:.2f}%"
)

print(
    f"Final Test Top-1        : "
    f"{test_metrics['accuracy']:.2f}%"
)

print(
    f"Final Test Top-5        : "
    f"{test_metrics['top5']:.2f}%"
)

print(
    f"Final Test Macro-F1     : "
    f"{test_metrics['f1']:.2f}%"
)

print()
print("Output directory:")
print(
    CHECKPOINT_DIR
)

print()
print("Files generated:")
print(
    "✓ best_model.pth"
)

print(
    "✓ last_checkpoint.pth"
)

print(
    "✓ epoch_XXX.pth"
)

print(
    "✓ training_history.csv"
)

print(
    "✓ classification_report.txt"
)

print(
    "✓ confusion_matrix.npy"
)

print()
print("=" * 70)