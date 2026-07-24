from datasets.mpii_train_dataset import MPIITrainDataset
from preprocessing.transforms import train_transform


dataset = MPIITrainDataset(

    image_dir="datasets/mpii/images",

    annotation_file=
    "datasets/mpii/annotations/mpii_human_pose_v1_u12_1.mat",

    transform=train_transform

)


print("Dataset Size:",len(dataset))


image, heatmap = dataset[0]


print("Image Shape:")
print(image.shape)


print("\nHeatmap Shape:")
print(heatmap.shape)