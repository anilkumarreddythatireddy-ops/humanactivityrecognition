from scipy.io import loadmat

annotation_path = "datasets/mpii/annotations/mpii_human_pose_v1_u12_1.mat"

mat = loadmat(annotation_path)

print("Keys in MAT file:")
print(mat.keys())

release = mat["RELEASE"]

print("\nType:", type(release))
print("Shape:", release.shape)
print("Data type:", release.dtype)