"""
Model definition for Car Model Recognition using transfer learning.
"""

import torch.nn as nn
from torchvision import models

def build_model(num_classes: int, backbone: str = "resnet50", freeze_backbone: bool = True):
    """
    Build a transfer-learning classifier on top of a pretrained backbone.

    Args:
        num_classes: number of output classes (brand+model, or brand+model+generation)
        backbone: one of "resnet50", "efficientnet_b0"
        freeze_backbone: if True, freeze all backbone weights except the new head

    Returns:
        torch.nn.Module
    """
    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if freeze_backbone:
           for param in model.parameters():
               param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
    )

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model


def unfreeze_last_n_layers(model, backbone: str, n: int = 20):
    """
    Unfreeze the last n parameter groups for fine-tuning after initial head training.
    Call this after a few epochs of head-only training for better accuracy.
    """
    params = list(model.parameters())
    for param in params[-n:]:
        param.requires_grad = True
    return model
