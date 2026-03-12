"""
augmentation.py — Data augmentation pipelines (torchvision-based).

Usage (from notebook):
    from src.augmentation import get_train_transform, get_val_transform
"""

from torchvision import transforms

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """
    Training augmentation pipeline.

    Applied ONLY to the train split; never to val or test.
    Normalization statistics are from ImageNet (used because we rely on
    pretrained backbones).
    """
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """
    Validation / test transform — deterministic, no augmentation.

    Uses the same normalization as training.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
