from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config import load_config, override_config
from .data import create_dataloaders
from .model import build_model
from .utils import dump_metrics, get_device, load_checkpoint, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN image classifier with PyTorch.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--epochs", type=int, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate.")
    parser.add_argument("--device", type=str, help="Preferred compute device (cuda/mps/cpu).")
    parser.add_argument("--model-dir", type=str, help="Directory for storing checkpoints.")
    parser.add_argument("--resume", action="store_true", help="Resume training from a saved checkpoint.")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume from.")
    parser.add_argument("--use-amp", action="store_true", help="Force enable AMP.")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP even if CUDA is available.")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    max_grad_norm: float,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    for inputs, targets in tqdm(loader, desc="Train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_acc += (outputs.argmax(dim=1) == targets).float().mean().item()
        num_batches += 1

    return running_loss / num_batches, running_acc / num_batches


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Eval", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_acc += (outputs.argmax(dim=1) == targets).float().mean().item()
            num_batches += 1

    return total_loss / num_batches, total_acc / num_batches


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "model_dir": args.model_dir,
        "use_amp": args.use_amp if args.use_amp or args.no_amp else None,
    }
    if args.no_amp:
        overrides["use_amp"] = False
    config = override_config(config, overrides)

    set_seed(config.seed)
    device = get_device(config.device)
    train_loader, val_loader, _, class_names = create_dataloaders(config)
    model = build_model(num_classes=config.num_classes or len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5, verbose=True)

    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and device.type == "cuda")

    start_epoch = 1
    best_val_acc = 0.0
    checkpoint_path = Path(config.model_dir) / f"{config.project_name}.pt"
    if args.resume and args.checkpoint:
        start_epoch, best_val_acc, _ = load_checkpoint(model, optimizer, args.checkpoint, device, scaler)
        checkpoint_path = Path(args.checkpoint)

    print(f"Training on device: {device}")
    for epoch in range(start_epoch, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            config.max_grad_norm,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:02d}/{config.epochs} "
            f"| train loss: {train_loss:.4f} acc: {train_acc:.4f} "
            f"| val loss: {val_loss:.4f} acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_val_acc,
                checkpoint_path,
                config,
                scaler=scaler if scaler.is_enabled() else None,
            )
            print(f"Saved new best model to {checkpoint_path}")
        dump_metrics(epoch, train_loss, val_loss, val_acc, Path(config.log_dir) / "metrics.jsonl")

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
