"""
train.py — Training loop helpers for PyTorch models.

Usage (from notebook):
    from src.train import train_one_epoch, validate, train_model
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluate on a validation or test loader. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler=None,
    num_epochs: int = 20,
    patience: int = 5,
    checkpoint_path: str | Path = "outputs/models/best_model.pth",
    device: str = "cpu",
    use_amp: bool = False,
):
    """
    Full training loop with early stopping and checkpoint saving.

    Returns a history dict with per-epoch train/val metrics.
    """
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    scaler = GradScaler() if use_amp and device != "cpu" else None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:>3}/{num_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Checkpoint saved → {checkpoint_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping after {epoch} epochs (patience={patience}).")
                break

    return history
