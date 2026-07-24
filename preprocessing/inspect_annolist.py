from scipy.io import loadmat

mat = loadmat(
    "datasets/mpii/annotations/mpii_human_pose_v1_u12_1.mat",
    struct_as_record=False,
    squeeze_me=True
)

release = mat["RELEASE"]

print(type(release))

print("\nAttributes inside RELEASE:")
print(release.__dict__.keys())

print("\nNumber of images:")
print(len(release.annolist))

print("\nFirst image object:")
print(release.annolist[0])

print("\nImage filename:")
print(release.annolist[0].image.name)