from preprocessing.mpii_dataset import MPIIDataset

dataset = MPIIDataset(
    image_dir="datasets/mpii/images",
    annotation_file="datasets/mpii/annotations/mpii_human_pose_v1_u12_1.mat"
)

print(len(dataset))