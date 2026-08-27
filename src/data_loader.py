"""
Dataset and DataLoader utilities for the Car Model Recognition project.

Expects data organized as:
    data/processed/train/<class_name>/*.jpg
    data/processed/val/<class_name>/*.jpg
    data/processed/test/<class_name>/*.jpg

Where <class_name> encodes brand_model (and optionally generation), e.g.
"bmw_3series" or "bmw_3series_2019-2023".
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet normalization stats (standard for pretrained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


def get_transforms(train: bool = True):
    """Return torchvision transforms for training or eval/inference."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(data_dir: str = "data/processed", batch_size: int = 32, num_workers: int = 2):
    """
    Build train/val/test DataLoaders from an ImageFolder-style directory structure.

    Returns:
        train_loader, val_loader, test_loader, class_names (list[str])
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(train=False))
    test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes
    return train_loader, val_loader, test_loader, class_names
