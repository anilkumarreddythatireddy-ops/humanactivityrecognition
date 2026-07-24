import os

mpii = "datasets/mpii/images"

print("MPII Images:", len(os.listdir(mpii)))

ucf = "datasets/ucf101"

classes = os.listdir(ucf)

print("Activities:", len(classes))

print(classes[:10])