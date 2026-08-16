# ============================================================
# STAGE 2
# UCF-101 VIDEO -> POSE SEQUENCE GENERATION
#
# VS CODE / WINDOWS VERSION
#
# Input:
#   datasets/ucf101/
#
# Model:
#   checkpoints/best_model.pth
#
# Output:
#   outputs/pose_sequences/
#
# Output sequence:
#   (16 frames, 16 joints, 2 coordinates)
#
# Resume supported:
#   Already generated .npy files are skipped.
# ============================================================

import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm


# ============================================================
# 1. PROJECT ROOT
# ============================================================

ROOT = r"C:\Users\NAVEEN REDDY\Desktop\hrr"

if not os.path.exists(ROOT):
    raise FileNotFoundError(
        f"Project root not found:\n{ROOT}\n\n"
        "Change ROOT at the top of this file."
    )

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ============================================================
# 2. IMPORT YOUR ViTPose MODEL
# ============================================================

try:

    from models.vit_pose import ViTPose

except Exception as e:

    print()
    print("=" * 70)
    print("ERROR IMPORTING ViTPose")
    print("=" * 70)
    print(e)
    print()
    print("Make sure this file exists:")
    print(
        os.path.join(
            ROOT,
            "models",
            "vit_pose.py"
        )
    )
    raise


# ============================================================
# 3. PATHS
# ============================================================

CHECKPOINT = os.path.join(
    ROOT,
    "checkpoints",
    "best_model.pth"
)

UCF_ROOT = os.path.join(
    ROOT,
    "datasets",
    "ucf101"
)

OUTPUT_ROOT = os.path.join(
    ROOT,
    "outputs",
    "pose_sequences"
)


# ============================================================
# 4. CONFIGURATION
# ============================================================

NUM_JOINTS = 16

IMAGE_SIZE = 256

HEATMAP_SIZE = 64

SEQUENCE_LENGTH = 16

# Number of frames sent to GPU/CPU at one time
FRAME_BATCH_SIZE = 16

# DataLoader is not used here because videos are processed
# individually. This keeps the code simple and resumable.

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 5. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)


# ============================================================
# 6. PRINT CONFIGURATION
# ============================================================

print()
print("=" * 70)
print("STAGE 2 - UCF-101 POSE SEQUENCE GENERATION")
print("=" * 70)

print()
print("Project Root :", ROOT)
print("Device       :", DEVICE)
print("Checkpoint   :", CHECKPOINT)
print("UCF Root     :", UCF_ROOT)
print("Output Root  :", OUTPUT_ROOT)

print()
print("Sequence length :", SEQUENCE_LENGTH)
print("Number of joints:", NUM_JOINTS)


# ============================================================
# 7. CHECK REQUIRED FILES/DIRECTORIES
# ============================================================

if not os.path.exists(CHECKPOINT):

    raise FileNotFoundError(
        "\nCheckpoint not found:\n"
        + CHECKPOINT
        + "\n\n"
        "Make sure best_model.pth exists inside checkpoints."
    )


if not os.path.exists(UCF_ROOT):

    raise FileNotFoundError(
        "\nUCF-101 directory not found:\n"
        + UCF_ROOT
        + "\n\n"
        "Expected structure:\n"
        "datasets/ucf101/<class>/<video>.avi"
    )


# ============================================================
# 8. LOAD ViTPose
# ============================================================

print()
print("=" * 70)
print("LOADING ViTPOSE")
print("=" * 70)

try:

    model = ViTPose(
        num_joints=NUM_JOINTS
    ).to(DEVICE)

except TypeError:

    # Compatibility with models that don't use num_joints
    model = ViTPose().to(DEVICE)


# ============================================================
# 9. LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)


# ============================================================
# 10. HANDLE DIFFERENT CHECKPOINT FORMATS
# ============================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        # Check whether the dictionary itself is a state dict
        state_dict = checkpoint

else:

    state_dict = checkpoint


# ============================================================
# 11. LOAD MODEL PARAMETERS
# ============================================================

try:

    model.load_state_dict(
        state_dict
    )

except RuntimeError as e:

    print()
    print("=" * 70)
    print("CHECKPOINT / MODEL MISMATCH")
    print("=" * 70)
    print(e)
    print()
    print(
        "The ViTPose architecture in models/vit_pose.py "
        "must match the architecture used during training."
    )
    raise


model.eval()


print()
print("Model Loaded Successfully")


if isinstance(checkpoint, dict):

    if "epoch" in checkpoint:

        print(
            "Checkpoint Epoch :",
            checkpoint["epoch"]
        )

    if "best_loss" in checkpoint:

        print(
            "Best Loss        :",
            checkpoint["best_loss"]
        )

    if "best_pck" in checkpoint:

        print(
            "Best PCK         :",
            checkpoint["best_pck"]
        )


# ============================================================
# 12. IMAGE NORMALIZATION
# ============================================================

MEAN = np.array(
    [
        0.485,
        0.456,
        0.406
    ],
    dtype=np.float32
)

STD = np.array(
    [
        0.229,
        0.224,
        0.225
    ],
    dtype=np.float32
)


# ============================================================
# 13. PREPROCESS FRAME
# ============================================================

def preprocess_frame(frame):
    """
    OpenCV BGR frame
        ->
    RGB
        ->
    256x256
        ->
    [0,1]
        ->
    ImageNet normalization
        ->
    CHW tensor
    """

    # BGR -> RGB

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Resize

    frame = cv2.resize(
        frame,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        interpolation=cv2.INTER_LINEAR
    )

    # uint8 -> float

    frame = frame.astype(
        np.float32
    ) / 255.0

    # Normalize

    frame = (
        frame - MEAN
    ) / STD

    # HWC -> CHW

    frame = np.transpose(
        frame,
        (2, 0, 1)
    )

    return torch.from_numpy(
        frame
    ).float()


# ============================================================
# 14. EXTRACT 16 FRAMES FROM VIDEO
# ============================================================

def extract_frames(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        return None

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:

        cap.release()

        return None

    # Uniform frame sampling

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    ).astype(
        np.int64
    )

    frames = []

    for index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        success, frame = cap.read()

        if not success:

            cap.release()

            return None

        processed = preprocess_frame(
            frame
        )

        frames.append(
            processed
        )

    cap.release()

    return torch.stack(
        frames,
        dim=0
    )


# ============================================================
# 15. HEATMAP -> COORDINATES
# ============================================================

def heatmaps_to_coordinates(
    heatmaps
):

    """
    Input:

        [T, J, H, W]

    Output:

        coordinates:
            [T, J, 2]

        confidence:
            [T, J]
    """

    T, J, H, W = heatmaps.shape

    # Flatten heatmaps

    flat = heatmaps.reshape(
        T,
        J,
        H * W
    )

    # Maximum location

    indices = flat.argmax(
        dim=-1
    )

    # Maximum confidence

    confidence = flat.amax(
        dim=-1
    )

    # X coordinate

    x = (
        indices % W
    ).float()

    # Y coordinate

    y = (
        indices // W
    ).float()

    coordinates = torch.stack(
        [
            x,
            y
        ],
        dim=-1
    )

    return (
        coordinates,
        confidence
    )


# ============================================================
# 16. NORMALIZE POSE
# ============================================================

def normalize_pose(
    coordinates
):

    """
    Input:

        [T, J, 2]

    Output:

        [T, J, 2]

    Center each frame around its mean
    joint position and scale by the
    maximum joint distance.
    """

    coordinates = coordinates.clone()

    T = coordinates.shape[0]

    for t in range(T):

        pose = coordinates[t]

        # Center

        center = pose.mean(
            dim=0,
            keepdim=True
        )

        pose = (
            pose - center
        )

        # Scale

        distances = torch.norm(
            pose,
            dim=1
        )

        scale = distances.max()

        if scale > 1e-6:

            pose = (
                pose / scale
            )

        coordinates[t] = pose

    return coordinates


# ============================================================
# 17. FIND ALL UCF-101 VIDEOS
# ============================================================

print()
print("=" * 70)
print("SEARCHING FOR UCF-101 VIDEOS")
print("=" * 70)

video_files = []

for root, dirs, files in os.walk(
    UCF_ROOT
):

    for file in files:

        if file.lower().endswith(
            (
                ".avi",
                ".mp4",
                ".mov"
            )
        ):

            video_files.append(
                os.path.join(
                    root,
                    file
                )
            )


video_files.sort()


print()
print(
    "Videos found :",
    len(video_files)
)


if len(video_files) == 0:

    raise RuntimeError(
        "No video files were found inside:\n"
        + UCF_ROOT
    )


# ============================================================
# 18. PROCESS VIDEOS
# ============================================================

successful = 0

failed = 0

skipped = 0


print()
print("=" * 70)
print("GENERATING POSE SEQUENCES")
print("=" * 70)

print()
print(
    "Already generated sequences will be skipped."
)

print()


for video_number, video_path in enumerate(
    tqdm(
        video_files,
        desc="Processing UCF-101"
    ),
    start=1
):

    try:

        # ----------------------------------------------------
        # Relative path
        # ----------------------------------------------------

        relative_path = os.path.relpath(
            video_path,
            UCF_ROOT
        )

        parts = relative_path.split(
            os.sep
        )

        # Expected:
        #
        # class/video.avi

        if len(parts) < 2:

            failed += 1

            continue

        class_name = parts[0]

        video_filename = os.path.basename(
            video_path
        )

        video_name = os.path.splitext(
            video_filename
        )[0]

        # ----------------------------------------------------
        # Create class output directory
        # ----------------------------------------------------

        class_output = os.path.join(
            OUTPUT_ROOT,
            class_name
        )

        os.makedirs(
            class_output,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Output files
        # ----------------------------------------------------

        output_file = os.path.join(
            class_output,
            video_name + ".npy"
        )

        confidence_file = os.path.join(
            class_output,
            video_name + "_confidence.npy"
        )

        # ----------------------------------------------------
        # RESUME SUPPORT
        # ----------------------------------------------------

        if os.path.exists(
            output_file
        ):

            skipped += 1

            continue

        # ----------------------------------------------------
        # Extract frames
        # ----------------------------------------------------

        frames = extract_frames(
            video_path
        )

        if frames is None:

            print()
            print(
                "Could not read video:"
            )
            print(
                video_path
            )

            failed += 1

            continue

        # ----------------------------------------------------
        # Move frames to device
        # ----------------------------------------------------

        frames = frames.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # ViTPose inference
        # ----------------------------------------------------

        heatmaps_list = []

        with torch.inference_mode():

            for start in range(
                0,
                SEQUENCE_LENGTH,
                FRAME_BATCH_SIZE
            ):

                end = min(
                    start + FRAME_BATCH_SIZE,
                    SEQUENCE_LENGTH
                )

                batch = frames[
                    start:end
                ]

                predictions = model(
                    batch
                )

                heatmaps_list.append(
                    predictions.detach()
                )

        # ----------------------------------------------------
        # Combine heatmaps
        # ----------------------------------------------------

        heatmaps = torch.cat(
            heatmaps_list,
            dim=0
        )

        # ----------------------------------------------------
        # Heatmaps -> coordinates
        # ----------------------------------------------------

        coordinates, confidence = (
            heatmaps_to_coordinates(
                heatmaps
            )
        )

        # ----------------------------------------------------
        # Normalize coordinates
        # ----------------------------------------------------

        normalized_coordinates = (
            normalize_pose(
                coordinates
            )
        )

        # ----------------------------------------------------
        # CPU
        # ----------------------------------------------------

        pose_sequence = (
            normalized_coordinates
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        confidence_sequence = (
            confidence
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        # ----------------------------------------------------
        # Verify shape
        # ----------------------------------------------------

        if pose_sequence.shape != (
            SEQUENCE_LENGTH,
            NUM_JOINTS,
            2
        ):

            print()
            print(
                "Unexpected pose shape:"
            )

            print(
                pose_sequence.shape
            )

            failed += 1

            continue

        # ----------------------------------------------------
        # SAVE POSE
        # ----------------------------------------------------

        np.save(
            output_file,
            pose_sequence
        )

        # ----------------------------------------------------
        # SAVE CONFIDENCE
        # ----------------------------------------------------

        np.save(
            confidence_file,
            confidence_sequence
        )

        successful += 1

    except Exception as e:

        failed += 1

        print()
        print("=" * 70)
        print("VIDEO PROCESSING ERROR")
        print("=" * 70)

        print(
            "Video:",
            video_path
        )

        print(
            "Error:",
            repr(e)
        )


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("STAGE 2 EXTRACTION COMPLETED")
print("=" * 70)

print()
print(
    "Total videos found     :",
    len(video_files)
)

print(
    "Successfully generated :",
    successful
)

print(
    "Already existed        :",
    skipped
)

print(
    "Failed                 :",
    failed
)

print()
print(
    "Output directory:"
)

print(
    OUTPUT_ROOT
)


# ============================================================
# 20. FIND GENERATED POSE FILES
# ============================================================

pose_files = []

for root, dirs, files in os.walk(
    OUTPUT_ROOT
):

    for file in files:

        if (
            file.endswith(".npy")
            and
            not file.endswith(
                "_confidence.npy"
            )
        ):

            pose_files.append(
                os.path.join(
                    root,
                    file
                )
            )


pose_files.sort()


# ============================================================
# 21. VERIFY GENERATED DATA
# ============================================================

print()
print("=" * 70)
print("VERIFYING GENERATED POSE DATA")
print("=" * 70)

print()
print(
    "Pose sequence files :",
    len(pose_files)
)


if len(pose_files) > 0:

    sample_file = pose_files[0]

    sample = np.load(
        sample_file
    )

    print()
    print(
        "Sample file:"
    )

    print(
        sample_file
    )

    print()
    print(
        "Shape     :",
        sample.shape
    )

    print(
        "Data type :",
        sample.dtype
    )

    print(
        "Min       :",
        float(sample.min())
    )

    print(
        "Max       :",
        float(sample.max())
    )

    print(
        "Mean      :",
        float(sample.mean())
    )

    print(
        "Std       :",
        float(sample.std())
    )

    print()

    expected_shape = (
        SEQUENCE_LENGTH,
        NUM_JOINTS,
        2
    )

    if sample.shape == expected_shape:

        print(
            "✓ SHAPE CHECK PASSED"
        )

    else:

        print(
            "✗ SHAPE CHECK FAILED"
        )

        print(
            "Expected:",
            expected_shape
        )


# ============================================================
# 22. SHOW FIRST FEW FILES
# ============================================================

print()
print("=" * 70)
print("SAMPLE GENERATED FILES")
print("=" * 70)

for file in pose_files[:10]:

    print(
        file
    )


# ============================================================
# 23. FINAL STATUS
# ============================================================

print()
print("=" * 70)
print("STAGE 2 STATUS")
print("=" * 70)

if len(pose_files) > 0:

    print()
    print(
        "✓ Pose sequences have been generated."
    )

    print(
        "✓ Each sequence should contain 16 frames."
    )

    print(
        "✓ Each frame contains 16 joints."
    )

    print(
        "✓ Each joint contains X,Y coordinates."
    )

    print(
        "✓ Resume support is enabled."
    )

    print()
    print(
        "Expected sequence:"
    )

    print(
        "(16, 16, 2)"
    )

    print()
    print(
        "Next stage: pose-sequence dataset preparation."
    )

else:

    print()
    print(
        "⚠ No pose sequences were generated."
    )

print()
print("=" * 70)
print("DONE")
print("=" * 70)