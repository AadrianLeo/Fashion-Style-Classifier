"""
models/cnn.py — EfficientNet-B0 and ResNet-50 wrappers for fine-tuning.

Usage (from notebook):
    from src.models.cnn import build_efficientnet_b0, build_resnet50
"""

import torch.nn as nn

NUM_CLASSES = 5


def build_efficientnet_b0(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """
    EfficientNet-B0 from timm with a custom classification head.

    Two-phase fine-tuning:
      Phase 1 — freeze backbone, train head only (5 epochs).
      Phase 2 — unfreeze all layers, train end-to-end (15 epochs, lower lr).
    """
    try:
        import timm
    except ImportError as e:
        raise ImportError("Install timm: pip install timm") from e

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def build_resnet50(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """
    ResNet-50 from torchvision with a custom classification head.
    """
    import torchvision.models as tvm

    weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
    model = tvm.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_backbone(model) -> None:
    """Freeze all parameters except the classification head (Phase 1)."""
    for name, param in model.named_parameters():
        if "classifier" not in name and "fc" not in name and "head" not in name:
            param.requires_grad = False


def unfreeze_all(model) -> None:
    """Unfreeze all parameters for end-to-end fine-tuning (Phase 2)."""
    for param in model.parameters():
        param.requires_grad = True
