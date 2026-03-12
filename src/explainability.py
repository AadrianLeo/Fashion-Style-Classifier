"""
explainability.py — Grad-CAM (CNN) and SHAP (classical) helper utilities.

Usage (from notebook):
    from src.explainability import gradcam_overlay, shap_summary
"""

import numpy as np


def gradcam_overlay(model, image_tensor, target_class: int, target_layer=None):
    """
    Compute a Grad-CAM heatmap for a single image tensor.

    Parameters
    ----------
    model : nn.Module
        A PyTorch CNN (EfficientNet or ResNet).
    image_tensor : torch.Tensor
        Shape (1, 3, H, W); preprocessed with ImageNet normalisation.
    target_class : int
        Class index to explain.
    target_layer : nn.Module, optional
        The convolutional layer to hook. Defaults to the last conv block.

    Returns
    -------
    heatmap : np.ndarray  shape (H, W), values in [0, 1]
    overlay : np.ndarray  shape (H, W, 3), RGB overlay
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as e:
        raise ImportError("Install grad-cam: pip install grad-cam") from e

    import torch

    if target_layer is None:
        # Try to auto-detect the last conv block
        layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        if not layers:
            raise ValueError("No Conv2d layer found in model.")
        target_layer = layers[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = None  # uses argmax by default; set to [ClassifierOutputTarget(target_class)] for specific class
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    heatmap = grayscale_cam[0]

    # Reconstruct a displayable RGB image from the normalised tensor
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * std + mean).clip(0, 1).astype(np.float32)

    overlay = show_cam_on_image(img_np, heatmap, use_rgb=True)
    return heatmap, overlay


def shap_summary(pipeline, X_sample, feature_names=None, max_display: int = 20):
    """
    Compute and plot a SHAP summary for a scikit-learn pipeline.

    Parameters
    ----------
    pipeline : sklearn.Pipeline
        A fitted classical ML pipeline (SVM / RF / KNN).
    X_sample : np.ndarray
        Feature matrix for a sample of the test set.
    feature_names : list of str, optional
        Names for each input feature.
    max_display : int
        Number of top features to display.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError("Install shap: pip install shap") from e

    # Transform through all steps except the final classifier
    X_transformed = X_sample
    for name, step in pipeline.steps[:-1]:
        X_transformed = step.transform(X_transformed)

    clf = pipeline.named_steps[pipeline.steps[-1][0]]
    explainer = shap.KernelExplainer(clf.predict_proba, shap.sample(X_transformed, 50))
    shap_values = explainer.shap_values(X_transformed, nsamples=100)
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, max_display=max_display)
    return shap_values
