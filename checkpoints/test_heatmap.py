from preprocessing.mpii_dataset import MPIIDataset
from preprocessing.transforms import train_transform
from preprocessing.heatmap import HeatmapGenerator

dataset = MPIIDataset(
    image_dir="datasets/mpii/images",
    annotation_file="datasets/mpii/annotations/mpii_human_pose_v1_u12_1.mat",
    transform=train_transform
)

image, joints, visibility, meta = dataset[0]

generator = HeatmapGenerator()

heatmaps = generator.generate(
    joints.numpy(),
    visibility.numpy()
)

print("Heatmap Shape:", heatmaps.shape)
print("Overall Max:", heatmaps.max())

for i in range(16):
    if visibility[i] > 0:
        print(f"Joint {i} max:", heatmaps[i].max())