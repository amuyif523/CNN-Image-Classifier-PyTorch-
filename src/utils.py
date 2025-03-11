from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from .config import TrainingConfig


def set_seed(seed: int) -> None:
    """Ensure deterministic-ish behavior across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(preferred: str | None = None) -> torch.device:
    """Return the preferred compute device, falling back gracefully."""
    if preferred:
        preferred = preferred.lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    path: str | Path,
    config: TrainingConfig,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> None:
    """Persist model/optimizer/scaler states."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config.__dict__,
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str | Path,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Tuple[int, float, Dict]:
    """Restore state from disk."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_metric = checkpoint.get("best_metric", 0.0)
    return start_epoch, best_metric, checkpoint.get("config", {})


def dump_metrics(epoch: int, train_loss: float, val_loss: float, val_acc: float, path: str | Path) -> None:
    """Append human-readable metrics for quick inspection."""
    payload = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
