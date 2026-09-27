"""
Model definitions for Agricultural Crop Classification.
Uses transfer learning (ResNet18 / MobileNetV3) as recommended for limited data.
"""

import torch
import torch.nn as nn
from torchvision import models


def create_model(
    model_name: str = "resnet18",
    num_classes: int = 6,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Create a transfer-learning model.

    Args:
        model_name: 'resnet18' | 'mobilenet_v3_small' | 'efficientnet_b0'
        num_classes: number of crop classes
        pretrained: use ImageNet weights
        freeze_backbone: freeze feature extractor (useful for very small data)
    """
    weights = "DEFAULT" if pretrained else None

    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes),
        )
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    elif model_name == "efficientnet_b0":
        try:
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            if freeze_backbone:
                for name, param in model.named_parameters():
                    if "classifier" not in name:
                        param.requires_grad = False
        except Exception:
            print("EfficientNet not available, falling back to ResNet18")
            return create_model("resnet18", num_classes, pretrained, freeze_backbone)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
