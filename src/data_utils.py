"""
data_utils.py — Dataset loading and splitting helpers.

Usage (from notebook):
    from src.data_utils import load_labels, make_datasets, CLASSES, SEED
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

SEED = 42
CLASSES = ["casual", "formal", "sporty", "streetwear", "vintage"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 224
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def set_seed(seed: int = SEED) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_labels(csv_path: str | Path = None) -> pd.DataFrame:
    """
    Load and normalise the Label Studio annotation CSV.

    Returns a DataFrame with columns: filename, label
    """
    if csv_path is None:
        # Prefer the canonical location; fall back to the legacy Dataset/ export
        canonical = DATA_DIR / "labels" / "labels_file.csv"
        legacy = DATA_DIR.parent / "Dataset" / "labels_file.csv"
        csv_path = canonical if canonical.exists() else legacy
    df = pd.read_csv(csv_path)
    # Rename Label Studio columns to a canonical schema
    rename = {"choice": "label", "image": "filename"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    # Keep only relevant columns
    df = df[["filename", "label"]].dropna()
    # Normalise label strings
    df["label"] = df["label"].str.strip().str.lower()
    # Drop rows whose label is not in the known class list
    df = df[df["label"].isin(CLASSES)].reset_index(drop=True)
    return df


def make_splits(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split.

    Leakage prevention:
    - Splits are created *before* any augmentation or normalisation.
    - Normalisation stats must be computed only on the train split.
    - The test split should be loaded only once, in the final report notebook.
    """
    test_size = 1.0 - train_size - val_size
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9, "Splits must sum to 1."

    train_df, temp_df = train_test_split(
        df, test_size=(1.0 - train_size), stratify=df["label"], random_state=seed
    )
    relative_val = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1.0 - relative_val), stratify=temp_df["label"], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels_dir: str | Path = None,
) -> None:
    """Save split manifests to data/labels/."""
    if labels_dir is None:
        labels_dir = DATA_DIR / "labels"
    Path(labels_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(Path(labels_dir) / "train_labels.csv", index=False)
    val_df.to_csv(Path(labels_dir) / "val_labels.csv", index=False)
    test_df.to_csv(Path(labels_dir) / "test_labels.csv", index=False)
    print(f"Splits saved to {labels_dir}")
    print(f"  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


def load_image(path: str | Path, size: int = IMG_SIZE) -> np.ndarray:
    """Load an image as a normalised NumPy array (C, H, W) in [0, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.array(img).astype(np.float32) / 255.0
