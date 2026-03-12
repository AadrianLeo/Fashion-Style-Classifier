"""
evaluate.py — Metric computation, confusion matrix plotting, and report generation.

Usage (from notebook):
    from src.evaluate import compute_metrics, plot_confusion_matrix, print_report
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

CLASSES = ["casual", "formal", "sporty", "streetwear", "vintage"]


def compute_metrics(y_true, y_pred, classes=None) -> dict:
    """
    Compute accuracy, weighted F1, macro F1, and per-class F1.

    Returns a dict suitable for JSON serialisation and MLflow logging.
    """
    if classes is None:
        classes = CLASSES
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class": {c: report[c] for c in classes if c in report},
        "classification_report": report,
    }


def print_report(y_true, y_pred, classes=None) -> None:
    """Print a formatted classification report."""
    if classes is None:
        classes = CLASSES
    print(classification_report(y_true, y_pred, target_names=classes))


def plot_confusion_matrix(
    y_true,
    y_pred,
    classes=None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot and optionally save a confusion matrix.

    Parameters
    ----------
    normalize : bool
        If True, normalise by true-label counts (row-normalised).
    save_path : str or Path, optional
        If provided, save the figure to this path.
    """
    if classes is None:
        classes = CLASSES
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Serialise a metrics dict to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {path}")


def bootstrap_ci(y_true, y_pred, metric_fn=None, n_boot: int = 1000, alpha: float = 0.05):
    """
    Compute bootstrap 95% confidence interval for a metric.

    Parameters
    ----------
    metric_fn : callable, optional
        Function with signature (y_true, y_pred) -> float.
        Defaults to weighted F1-score.
    """
    if metric_fn is None:
        metric_fn = lambda yt, yp: f1_score(yt, yp, average="weighted")
    rng = np.random.default_rng(42)
    n = len(y_true)
    scores = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    point = metric_fn(y_true, y_pred)
    return {"point": point, "lower": lower, "upper": upper, "ci": f"[{lower:.3f}, {upper:.3f}]"}
