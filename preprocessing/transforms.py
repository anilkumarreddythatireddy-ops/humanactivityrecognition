from torchvision import transforms
from configs.config import Config

train_transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize(
        (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=Config.MEAN,
        std=Config.STD
    )
])