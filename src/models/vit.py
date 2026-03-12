"""
models/vit.py — Vision Transformer wrappers (ViT-B/16, Swin-T) for fine-tuning.

Usage (from notebook):
    from src.models.vit import build_vit_b16, build_swin_tiny
"""

NUM_CLASSES = 5


def build_vit_b16(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """
    ViT-B/16 from timm, pretrained on ImageNet-21k.

    With ~86M parameters this is the largest model in the suite.
    Consider using a linear probe (freeze=True) if GPU memory is limited.
    """
    try:
        import timm
    except ImportError as e:
        raise ImportError("Install timm: pip install timm") from e

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def build_swin_tiny(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """
    Swin Transformer Tiny from timm, pretrained on ImageNet-1k.

    A lighter alternative to ViT-B/16 (~28M params).
    """
    try:
        import timm
    except ImportError as e:
        raise ImportError("Install timm: pip install timm") from e

    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
