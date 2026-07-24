from preprocessing.mpii_dataset import MPIIDataset
from preprocessing.transforms import train_transform

dataset = MPIIDataset(
    image_dir="datasets/mpii/images",
    annotation_file="datasets/mpii/annotations/mpii_human_pose_v1_u12_1.mat",
    transform=train_transform
)

image, joints, visibility, meta = dataset[0]

print("Image shape:", image.shape)

print("\nJoints:")
print(joints)

print("\nVisibility:")
print(visibility)

print("\nVisible joints:")

for i in range(16):
    if visibility[i] > 0:
        print(
            i,
            joints[i][0].item(),
            joints[i][1].item()
        )