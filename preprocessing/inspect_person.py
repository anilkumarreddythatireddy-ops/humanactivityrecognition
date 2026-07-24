from scipy.io import loadmat

mat = loadmat(
    "datasets/mpii/annotations/mpii_human_pose_v1_u12_1.mat",
    struct_as_record=False,
    squeeze_me=True
)

release = mat["RELEASE"]

for ann in release.annolist:

    if hasattr(ann, "annorect"):

        rect = ann.annorect

        if isinstance(rect, list):
            rect = rect[0]

        if hasattr(rect, "annopoints"):

            print("Image:", ann.image.name)

            print("\nPerson fields:")
            print(rect.__dict__.keys())

            print("\nPoint fields:")
            print(rect.annopoints.point[0].__dict__.keys())

            break