# ============================================================
# STAGE 3 - POSE SEQUENCE DATASET PREPARATION
# VS CODE VERSION
# ============================================================

import os
import re
import csv
import random
import numpy as np
from collections import Counter


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Example:
# C:\Users\NAVEEN REDDY\Desktop\hrr


# ============================================================
# PATHS
# ============================================================

POSE_ROOT = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "pose_sequences",
    "UCF101"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "pose_sequence"
)

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)

EXPECTED_SHAPE = (
    16,
    16,
    2
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STAGE 3 - POSE SEQUENCE DATASET PREPARATION")
print("=" * 70)

print()

print("Project Root :", PROJECT_ROOT)
print("Pose Root    :", POSE_ROOT)
print("Output Root  :", OUTPUT_ROOT)


# ============================================================
# CHECK POSE DIRECTORY
# ============================================================

if not os.path.exists(POSE_ROOT):

    raise FileNotFoundError(
        "\nPose sequence directory was not found:\n"
        + POSE_ROOT
        + "\n\n"
        "Make sure Stage 2 pose extraction has been completed."
    )


# ============================================================
# FIND ALL POSE FILES
# ============================================================

print()
print("=" * 70)
print("SEARCHING POSE SEQUENCES")
print("=" * 70)

pose_files = []

for root, dirs, files in os.walk(POSE_ROOT):

    for filename in files:

        if not filename.endswith(".npy"):
            continue

        # Ignore confidence files

        if filename.endswith(
            "_confidence.npy"
        ):
            continue

        full_path = os.path.join(
            root,
            filename
        )

        pose_files.append(
            full_path
        )


pose_files.sort()


print()
print(
    "Pose sequences found :",
    len(pose_files)
)


if len(pose_files) == 0:

    raise RuntimeError(
        "No .npy pose sequences were found."
    )


# ============================================================
# CLASS NAME EXTRACTION
# ============================================================

def get_class_name(filepath):

    filename = os.path.basename(
        filepath
    )

    name = os.path.splitext(
        filename
    )[0]

    # Example:
    #
    # v_ApplyEyeMakeup_g01_c01
    #
    # Remove v_

    if name.startswith("v_"):

        name = name[2:]


    # Remove:
    #
    # _g01_c01

    match = re.match(
        r"(.+)_g\d+_c\d+$",
        name
    )

    if match:

        return match.group(1)


    # Fallback

    parts = name.split("_")

    if len(parts) >= 3:

        return "_".join(
            parts[:-2]
        )


    return name


# ============================================================
# VALIDATE SEQUENCES
# ============================================================

print()
print("=" * 70)
print("VALIDATING POSE SEQUENCES")
print("=" * 70)

records = []

invalid_files = []

shape_errors = []


for index, pose_file in enumerate(
    pose_files
):

    try:

        data = np.load(
            pose_file,
            mmap_mode="r"
        )

        # Expected:
        #
        # 16 frames
        # 16 joints
        # 2 coordinates

        if data.shape != EXPECTED_SHAPE:

            shape_errors.append(
                (
                    pose_file,
                    data.shape
                )
            )

            continue


        # Check NaN / Inf

        if not np.isfinite(
            data
        ).all():

            invalid_files.append(
                pose_file
            )

            continue


        class_name = get_class_name(
            pose_file
        )


        records.append(
            {
                "path": os.path.abspath(
                    pose_file
                ),
                "class_name": class_name
            }
        )


    except Exception as error:

        invalid_files.append(
            pose_file
        )


print()

print(
    "Valid sequences  :",
    len(records)
)

print(
    "Shape errors     :",
    len(shape_errors)
)

print(
    "Invalid files    :",
    len(invalid_files)
)


# ============================================================
# CLASS LIST
# ============================================================

classes = sorted(
    set(
        record["class_name"]
        for record in records
    )
)


class_to_index = {
    class_name: index
    for index, class_name in enumerate(
        classes
    )
}


print()
print("=" * 70)
print("CLASS INFORMATION")
print("=" * 70)

print()

print(
    "Number of classes :",
    len(classes)
)


# ============================================================
# SAVE CLASSES
# ============================================================

classes_file = os.path.join(
    OUTPUT_ROOT,
    "classes.txt"
)


with open(
    classes_file,
    "w",
    encoding="utf-8"
) as file:

    for index, class_name in enumerate(
        classes
    ):

        file.write(
            f"{index},{class_name}\n"
        )


print()

print(
    "Classes saved:"
)

print(
    classes_file
)


# ============================================================
# ADD LABELS
# ============================================================

for record in records:

    record["label"] = class_to_index[
        record["class_name"]
    ]


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_counts = Counter(
    record["class_name"]
    for record in records
)


print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print()

for class_name in classes:

    print(
        f"{class_name:35s} : "
        f"{class_counts[class_name]}"
    )


# ============================================================
# EXTRACT GROUP ID
# ============================================================

def get_group_id(filepath):

    filename = os.path.basename(
        filepath
    )

    match = re.search(
        r"_g(\d+)_c\d+",
        filename
    )

    if match:

        return int(
            match.group(1)
        )

    return None


# ============================================================
# GROUP RECORDS
# ============================================================

print()
print("=" * 70)
print("GROUP-AWARE DATASET SPLIT")
print("=" * 70)


grouped = {}


for record in records:

    group_id = get_group_id(
        record["path"]
    )

    key = (
        record["class_name"],
        group_id
    )

    if key not in grouped:

        grouped[key] = []

    grouped[key].append(
        record
    )


# ============================================================
# TRAIN / VALIDATION / TEST
#
# 70% TRAIN
# 15% VALIDATION
# 15% TEST
#
# Groups are kept together.
# ============================================================

train_records = []

val_records = []

test_records = []


for class_name in classes:

    groups = []

    for (
        current_class,
        group_id
    ) in grouped.keys():

        if current_class == class_name:

            groups.append(
                group_id
            )


    groups = sorted(
        set(groups)
    )

    random.shuffle(
        groups
    )


    number_of_groups = len(
        groups
    )


    if number_of_groups >= 3:

        train_count = max(
            1,
            int(
                0.70 *
                number_of_groups
            )
        )

        val_count = max(
            1,
            int(
                0.15 *
                number_of_groups
            )
        )

        test_count = (
            number_of_groups
            - train_count
            - val_count
        )


        if test_count < 1:

            test_count = 1

            train_count = max(
                1,
                train_count - 1
            )


    else:

        train_count = max(
            1,
            number_of_groups - 1
        )

        val_count = 0

        test_count = (
            number_of_groups
            - train_count
        )


    train_groups = groups[
        :train_count
    ]

    val_groups = groups[
        train_count:
        train_count + val_count
    ]

    test_groups = groups[
        train_count + val_count:
    ]


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for group_id in train_groups:

        key = (
            class_name,
            group_id
        )

        train_records.extend(
            grouped[key]
        )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    for group_id in val_groups:

        key = (
            class_name,
            group_id
        )

        val_records.extend(
            grouped[key]
        )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    for group_id in test_groups:

        key = (
            class_name,
            group_id
        )

        test_records.extend(
            grouped[key]
        )


# ============================================================
# SHUFFLE DATA
# ============================================================

random.shuffle(
    train_records
)

random.shuffle(
    val_records
)

random.shuffle(
    test_records
)


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    records,
    filename
):

    output_file = os.path.join(
        OUTPUT_ROOT,
        filename
    )


    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "path",
                "label",
                "class_name"
            ]
        )


        for record in records:

            writer.writerow(
                [
                    record["path"],
                    record["label"],
                    record["class_name"]
                ]
            )


    return output_file


train_csv = save_csv(
    train_records,
    "train.csv"
)

val_csv = save_csv(
    val_records,
    "val.csv"
)

test_csv = save_csv(
    test_records,
    "test.csv"
)


# ============================================================
# SAVE INVALID FILE LIST
# ============================================================

failed_file = os.path.join(
    OUTPUT_ROOT,
    "invalid_or_failed_sequences.txt"
)


with open(
    failed_file,
    "w",
    encoding="utf-8"
) as file:

    for path in invalid_files:

        file.write(
            path + "\n"
        )


    for path, shape in shape_errors:

        file.write(
            f"{path} | shape={shape}\n"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("STAGE 3 DATASET PREPARATION COMPLETED")
print("=" * 70)

print()

print(
    "Total valid sequences :",
    len(records)
)

print(
    "Training sequences    :",
    len(train_records)
)

print(
    "Validation sequences  :",
    len(val_records)
)

print(
    "Test sequences        :",
    len(test_records)
)

print()

print(
    "Number of classes     :",
    len(classes)
)


print()
print(
    "Train CSV:"
)

print(
    train_csv
)


print()
print(
    "Validation CSV:"
)

print(
    val_csv
)


print()
print(
    "Test CSV:"
)

print(
    test_csv
)


print()
print(
    "Classes:"
)

print(
    classes_file
)


# ============================================================
# DATA LEAKAGE CHECK
# ============================================================

train_paths = set(
    record["path"]
    for record in train_records
)

val_paths = set(
    record["path"]
    for record in val_records
)

test_paths = set(
    record["path"]
    for record in test_records
)


train_val_overlap = (
    train_paths &
    val_paths
)

train_test_overlap = (
    train_paths &
    test_paths
)

val_test_overlap = (
    val_paths &
    test_paths
)


print()
print("=" * 70)
print("DATA LEAKAGE CHECK")
print("=" * 70)

print()

print(
    "Train ∩ Validation :",
    len(train_val_overlap)
)

print(
    "Train ∩ Test       :",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test  :",
    len(val_test_overlap)
)


if (
    len(train_val_overlap) == 0
    and
    len(train_test_overlap) == 0
    and
    len(val_test_overlap) == 0
):

    print()
    print(
        "✓ NO PATH OVERLAP DETECTED"
    )

else:

    print()
    print(
        "⚠ OVERLAP DETECTED"
    )


# ============================================================
# SPLIT DISTRIBUTION
# ============================================================

def get_distribution(
    records
):

    return Counter(
        record["class_name"]
        for record in records
    )


train_distribution = get_distribution(
    train_records
)

val_distribution = get_distribution(
    val_records
)

test_distribution = get_distribution(
    test_records
)


missing_train = [
    class_name
    for class_name in classes
    if train_distribution[class_name] == 0
]

missing_val = [
    class_name
    for class_name in classes
    if val_distribution[class_name] == 0
]

missing_test = [
    class_name
    for class_name in classes
    if test_distribution[class_name] == 0
]


print()
print("=" * 70)
print("SPLIT CLASS CHECK")
print("=" * 70)

print()

print(
    "Classes missing from train :",
    len(missing_train)
)

print(
    "Classes missing from val   :",
    len(missing_val)
)

print(
    "Classes missing from test  :",
    len(missing_test)
)


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)
print("STAGE 3 STATUS")
print("=" * 70)

print()

print("✓ Pose sequences loaded")
print("✓ Shape validation completed")
print("✓ UCF-101 class labels extracted")
print("✓ Class index mapping created")
print("✓ Train CSV created")
print("✓ Validation CSV created")
print("✓ Test CSV created")
print("✓ Data leakage checked")
print("✓ Failed sequence list created")

print()

print(
    "Dataset directory:"
)

print(
    OUTPUT_ROOT
)

print()

print("=" * 70)
print("READY FOR TRANSFORMER + VAE")
print("=" * 70)