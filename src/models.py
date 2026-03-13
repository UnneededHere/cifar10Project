"""
Model definitions: CIFAR-10-adapted ResNet-18 and Tiny ViT.
"""

import timm
import torch.nn as nn
from torchvision import models


def get_resnet18(num_classes=10):
    """ResNet-18 adapted for 32x32 CIFAR-10 images.

    Modifications from standard ImageNet ResNet-18:
      - conv1: 3x3 kernel, stride 1, padding 1 (was 7x7, stride 2, padding 3)
      - Removes the initial MaxPool layer
    """
    model = models.resnet18(weights=None, num_classes=num_classes)

    # Replace the aggressive 7x7 stem with a gentle 3x3
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Remove maxpool — identity pass-through
    model.maxpool = nn.Identity()

    return model


def get_tiny_vit(num_classes=10):
    """Tiny Vision Transformer sized for 32x32 CIFAR-10 images.

    Uses timm's vit_tiny_patch16_224 as the backbone, reconfigured:
      - img_size=32, patch_size=4  →  (32/4)^2 = 64 tokens
      - 10 output classes
    """
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        img_size=32,
        patch_size=4,
        num_classes=num_classes,
    )
    return model


# Convenience dispatcher
_MODEL_MAP = {
    "resnet": get_resnet18,
    "vit": get_tiny_vit,
}


def get_model(name, num_classes=10):
    """Return a model by name ('resnet' or 'vit')."""
    if name not in _MODEL_MAP:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(_MODEL_MAP)}")
    return _MODEL_MAP[name](num_classes=num_classes)
