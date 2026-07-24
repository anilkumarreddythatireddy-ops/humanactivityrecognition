from scipy.io import loadmat

mat = loadmat("datasets/mpii/annotations/mpii_human_pose_v1_u12_1.mat")

print(mat.keys())