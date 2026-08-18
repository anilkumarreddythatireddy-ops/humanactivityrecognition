# ============================================================
# STAGE 4 - TRANSFORMER + VAE ACTIVITY RECOGNITION
# UCF-101 POSE SEQUENCES
# ============================================================

import os
import csv
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "pose_sequence"
)

CHECKPOINT_ROOT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "activity"
)

os.makedirs(
    CHECKPOINT_ROOT,
    exist_ok=True
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


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


NUM_FRAMES = 16
NUM_JOINTS = 16
COORDINATES = 2

INPUT_DIM = (
    NUM_JOINTS *
    COORDINATES
)

NUM_CLASSES = 101


# Model

D_MODEL = 256

NHEAD = 8

NUM_LAYERS = 4

DIM_FEEDFORWARD = 512

DROPOUT = 0.1

LATENT_DIM = 128


# Training

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

BETA_KL = 0.001

BETA_RECON = 0.1


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STAGE 4 - TRANSFORMER + VAE ACTIVITY RECOGNITION")
print("=" * 70)

print()

print("Project Root :", PROJECT_ROOT)

print("Device       :", DEVICE)

print("Train CSV    :", TRAIN_CSV)

print("Validation   :", VAL_CSV)

print("Test CSV     :", TEST_CSV)

print()


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    CLASSES_FILE
]


for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            "\nRequired file not found:\n"
            + file_path
        )


# ============================================================
# LOAD CLASSES
# ============================================================

classes = []

with open(
    CLASSES_FILE,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        line = line.strip()

        if not line:
            continue

        parts = line.split(
            ",",
            1
        )

        if len(parts) == 2:

            classes.append(
                parts[1]
            )

        else:

            classes.append(
                parts[0]
            )


NUM_CLASSES = len(classes)


print(
    "Number of classes :",
    NUM_CLASSES
)

print()


# ============================================================
# DATASET
# ============================================================

class PoseSequenceDataset(
    Dataset
):

    def __init__(
        self,
        csv_file
    ):

        self.samples = []

        with open(
            csv_file,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(
                file
            )

            for row in reader:

                path = row["path"]

                label = int(
                    row["label"]
                )

                self.samples.append(
                    (
                        path,
                        label
                    )
                )


    def __len__(
        self
    ):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):

        path, label = (
            self.samples[index]
        )

        pose = np.load(
            path
        ).astype(
            np.float32
        )


        # Expected:
        #
        # (16, 16, 2)

        if pose.shape != (
            NUM_FRAMES,
            NUM_JOINTS,
            COORDINATES
        ):

            raise ValueError(
                f"Invalid pose shape "
                f"{pose.shape} in {path}"
            )


        # Flatten joints:
        #
        # (16,16,2)
        #
        # →
        #
        # (16,32)

        pose = pose.reshape(
            NUM_FRAMES,
            INPUT_DIM
        )


        pose = torch.from_numpy(
            pose
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )


        return pose, label


# ============================================================
# LOAD DATASETS
# ============================================================

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


print()

print(
    "Training samples   :",
    len(train_dataset)
)

print(
    "Validation samples :",
    len(val_dataset)
)

print(
    "Test samples       :",
    len(test_dataset)
)

print()


# ============================================================
# DATALOADERS
# ============================================================

PIN_MEMORY = (
    DEVICE.type == "cuda"
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=PIN_MEMORY
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=PIN_MEMORY
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=PIN_MEMORY
)


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(
    nn.Module
):

    def __init__(
        self,
        d_model,
        max_len=100
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

        pe = pe.unsqueeze(
            0
        )

        self.register_buffer(
            "pe",
            pe
        )


    def forward(
        self,
        x
    ):

        return (
            x
            +
            self.pe[
                :,
                :x.size(1)
            ]
        )


# ============================================================
# TRANSFORMER + VAE MODEL
# ============================================================

class PoseTransformerVAE(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        num_classes,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        latent_dim=128,
        dropout=0.1
    ):

        super().__init__()


        # ----------------------------------------------------
        # INPUT EMBEDDING
        # ----------------------------------------------------

        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )


        # ----------------------------------------------------
        # POSITIONAL ENCODING
        # ----------------------------------------------------

        self.position_encoding = PositionalEncoding(
            d_model,
            max_len=NUM_FRAMES
        )


        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True
            )
        )


        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
        )


        # ----------------------------------------------------
        # LATENT REPRESENTATION
        # ----------------------------------------------------

        self.mu_layer = nn.Linear(
            d_model,
            latent_dim
        )

        self.logvar_layer = nn.Linear(
            d_model,
            latent_dim
        )


        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                latent_dim,
                256
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                256,
                num_classes
            )

        )


        # ----------------------------------------------------
        # VAE DECODER
        # ----------------------------------------------------

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
                256
            ),

            nn.GELU(),

            nn.Linear(
                256,
                NUM_FRAMES * INPUT_DIM
            )

        )


    # ========================================================
    # REPARAMETERIZATION
    # ========================================================

    def reparameterize(
        self,
        mu,
        logvar
    ):

        std = torch.exp(
            0.5 * logvar
        )

        epsilon = torch.randn_like(
            std
        )

        return (
            mu
            +
            epsilon * std
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x
    ):

        # x:
        #
        # (B,16,32)

        x = self.input_projection(
            x
        )

        x = self.position_encoding(
            x
        )

        x = self.transformer(
            x
        )


        # Temporal average

        representation = x.mean(
            dim=1
        )


        # VAE

        mu = self.mu_layer(
            representation
        )

        logvar = self.logvar_layer(
            representation
        )

        z = self.reparameterize(
            mu,
            logvar
        )


        # Classification

        logits = self.classifier(
            z
        )


        # Reconstruction

        reconstruction = self.decoder(
            z
        )


        reconstruction = reconstruction.reshape(
            -1,
            NUM_FRAMES,
            INPUT_DIM
        )


        return (
            logits,
            reconstruction,
            mu,
            logvar
        )


# ============================================================
# CREATE MODEL
# ============================================================

print("=" * 70)
print("CREATING MODEL")
print("=" * 70)

model = PoseTransformerVAE(
    input_dim=INPUT_DIM,
    num_classes=NUM_CLASSES,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    latent_dim=LATENT_DIM,
    dropout=DROPOUT
)


model = model.to(
    DEVICE
)


total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)


print()

print(
    "Model parameters :",
    f"{total_parameters:,}"
)

print()

print(
    "✓ Transformer created"
)

print(
    "✓ VAE encoder created"
)

print(
    "✓ VAE decoder created"
)

print(
    "✓ 101-class classifier created"
)

print()


# ============================================================
# LOSS FUNCTIONS
# ============================================================

classification_loss_function = (
    nn.CrossEntropyLoss()
)

reconstruction_loss_function = (
    nn.MSELoss()
)


def calculate_kl_loss(
    mu,
    logvar
):

    kl = -0.5 * (
        1
        +
        logvar
        -
        mu.pow(2)
        -
        logvar.exp()
    )

    return kl.mean()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# AMP
# ============================================================

use_amp = (
    DEVICE.type == "cuda"
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# CHECKPOINT VARIABLES
# ============================================================

start_epoch = 1

best_val_accuracy = 0.0

best_val_loss = float(
    "inf"
)


last_checkpoint = os.path.join(
    CHECKPOINT_ROOT,
    "last_checkpoint.pth"
)

best_checkpoint = os.path.join(
    CHECKPOINT_ROOT,
    "best_model.pth"
)


# ============================================================
# RESUME
# ============================================================

if os.path.exists(
    last_checkpoint
):

    print("=" * 70)
    print("LOADING LAST CHECKPOINT")
    print("=" * 70)

    checkpoint = torch.load(
        last_checkpoint,
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


    start_epoch = (
        checkpoint["epoch"]
        + 1
    )

    best_val_accuracy = (
        checkpoint.get(
            "best_val_accuracy",
            0.0
        )
    )

    best_val_loss = (
        checkpoint.get(
            "best_val_loss",
            float("inf")
        )
    )


    print()

    print(
        "Checkpoint epoch :",
        checkpoint["epoch"]
    )

    print(
        "Starting epoch   :",
        start_epoch
    )

    print(
        "Best validation accuracy :",
        f"{best_val_accuracy:.2f}%"
    )

else:

    print(
        "No previous checkpoint found."
    )

    print(
        "Starting from Epoch 1."
    )


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    total_classification = 0.0

    total_reconstruction = 0.0

    total_kl = 0.0

    correct = 0

    total = 0


    for batch_index, (
        poses,
        labels
    ) in enumerate(
        train_loader
    ):

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


        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp
        ):

            logits, reconstruction, mu, logvar = (
                model(poses)
            )


            classification_loss = (
                classification_loss_function(
                    logits,
                    labels
                )
            )


            reconstruction_loss = (
                reconstruction_loss_function(
                    reconstruction,
                    poses
                )
            )


            kl_loss = (
                calculate_kl_loss(
                    mu,
                    logvar
                )
            )


            loss = (
                classification_loss
                +
                BETA_RECON *
                reconstruction_loss
                +
                BETA_KL *
                kl_loss
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


        total_loss += (
            loss.item()
        )

        total_classification += (
            classification_loss.item()
        )

        total_reconstruction += (
            reconstruction_loss.item()
        )

        total_kl += (
            kl_loss.item()
        )


        predictions = (
            torch.argmax(
                logits,
                dim=1
            )
        )


        correct += (
            predictions == labels
        ).sum().item()

        total += (
            labels.size(0)
        )


        if (
            batch_index % 50 == 0
        ):

            current_accuracy = (
                100.0
                *
                correct
                /
                max(total, 1)
            )


            print(
                f"Batch "
                f"{batch_index:4d}/"
                f"{len(train_loader):4d}"
                f" | Loss: "
                f"{loss.item():.5f}"
                f" | Acc: "
                f"{current_accuracy:.2f}%"
            )


    number_of_batches = len(
        train_loader
    )


    return (
        total_loss / number_of_batches,
        total_classification / number_of_batches,
        total_reconstruction / number_of_batches,
        total_kl / number_of_batches,
        100.0 * correct / total
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    loader
):

    model.eval()

    total_loss = 0.0

    total_classification = 0.0

    total_reconstruction = 0.0

    total_kl = 0.0

    correct = 0

    total = 0


    for poses, labels in loader:

        poses = poses.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp
        ):

            logits, reconstruction, mu, logvar = (
                model(poses)
            )


            classification_loss = (
                classification_loss_function(
                    logits,
                    labels
                )
            )


            reconstruction_loss = (
                reconstruction_loss_function(
                    reconstruction,
                    poses
                )
            )


            kl_loss = (
                calculate_kl_loss(
                    mu,
                    logvar
                )
            )


            loss = (
                classification_loss
                +
                BETA_RECON *
                reconstruction_loss
                +
                BETA_KL *
                kl_loss
            )


        total_loss += (
            loss.item()
        )

        total_classification += (
            classification_loss.item()
        )

        total_reconstruction += (
            reconstruction_loss.item()
        )

        total_kl += (
            kl_loss.item()
        )


        predictions = (
            torch.argmax(
                logits,
                dim=1
            )
        )


        correct += (
            predictions == labels
        ).sum().item()

        total += (
            labels.size(0)
        )


    batches = len(loader)


    return (
        total_loss / batches,
        total_classification / batches,
        total_reconstruction / batches,
        total_kl / batches,
        100.0 * correct / total
    )


# ============================================================
# TRAINING LOOP
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print()

for epoch in range(
    start_epoch,
    EPOCHS + 1
):

    epoch_start = time.time()


    print()
    print("=" * 70)

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    (
        train_loss,
        train_cls,
        train_recon,
        train_kl,
        train_accuracy
    ) = train_one_epoch()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    (
        val_loss,
        val_cls,
        val_recon,
        val_kl,
        val_accuracy
    ) = evaluate(
        val_loader
    )


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step()


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    epoch_time = (
        time.time()
        -
        epoch_start
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()

    print(
        f"Epoch {epoch} Completed"
    )

    print(
        f"Training Loss       : "
        f"{train_loss:.6f}"
    )

    print(
        f"Training Accuracy   : "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Training CE Loss    : "
        f"{train_cls:.6f}"
    )

    print(
        f"Training Recon Loss : "
        f"{train_recon:.6f}"
    )

    print(
        f"Training KL Loss    : "
        f"{train_kl:.6f}"
    )

    print()

    print(
        f"Validation Loss     : "
        f"{val_loss:.6f}"
    )

    print(
        f"Validation Accuracy : "
        f"{val_accuracy:.2f}%"
    )

    print(
        f"Validation CE Loss  : "
        f"{val_cls:.6f}"
    )

    print(
        f"Validation Recon    : "
        f"{val_recon:.6f}"
    )

    print(
        f"Validation KL       : "
        f"{val_kl:.6f}"
    )

    print()

    print(
        f"Learning Rate       : "
        f"{current_lr:.8f}"
    )

    print(
        f"Time                : "
        f"{epoch_time / 60:.2f} min"
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    is_best = (
        val_accuracy
        >
        best_val_accuracy
    )


    if is_best:

        best_val_accuracy = (
            val_accuracy
        )

        best_val_loss = (
            val_loss
        )


        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "best_val_accuracy":
                    best_val_accuracy,

                "best_val_loss":
                    best_val_loss,

                "classes":
                    classes,

                "config":
                    {
                        "num_frames":
                            NUM_FRAMES,

                        "num_joints":
                            NUM_JOINTS,

                        "input_dim":
                            INPUT_DIM,

                        "num_classes":
                            NUM_CLASSES,

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
            best_checkpoint
        )


        print()

        print(
            "✓ BEST MODEL UPDATED"
        )

        print(
            "Best Validation Accuracy : "
            f"{best_val_accuracy:.2f}%"
        )


    # --------------------------------------------------------
    # LAST CHECKPOINT
    # --------------------------------------------------------

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "best_val_accuracy":
                best_val_accuracy,

            "best_val_loss":
                best_val_loss,

            "classes":
                classes,

            "config":
                {
                    "num_frames":
                        NUM_FRAMES,

                    "num_joints":
                        NUM_JOINTS,

                    "input_dim":
                        INPUT_DIM,

                    "num_classes":
                        NUM_CLASSES,

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
        last_checkpoint
    )


    print(
        "Checkpoint Saved:"
    )

    print(
        last_checkpoint
    )


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print()

print(
    "Best Validation Accuracy :",
    f"{best_val_accuracy:.2f}%"
)

print(
    "Best Model :",
    best_checkpoint
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL FOR TEST")
print("=" * 70)


best = torch.load(
    best_checkpoint,
    map_location=DEVICE
)


model.load_state_dict(
    best["model_state_dict"]
)


# ============================================================
# TEST
# ============================================================

(
    test_loss,
    test_cls,
    test_recon,
    test_kl,
    test_accuracy
) = evaluate(
    test_loader
)


print()

print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print()

print(
    f"Test Loss       : "
    f"{test_loss:.6f}"
)

print(
    f"Test Accuracy   : "
    f"{test_accuracy:.2f}%"
)

print(
    f"Test CE Loss    : "
    f"{test_cls:.6f}"
)

print(
    f"Test Recon Loss : "
    f"{test_recon:.6f}"
)

print(
    f"Test KL Loss    : "
    f"{test_kl:.6f}"
)

print()

print("=" * 70)
print("STAGE 4 COMPLETED")
print("=" * 70)