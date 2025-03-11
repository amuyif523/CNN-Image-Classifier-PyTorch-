from __future__ import annotations

import argparse

import torch
from torch import nn
from tqdm import tqdm

from .config import load_config, override_config
from .data import create_dataloaders
from .model import build_model
from .utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN classifier.")
    parser.add_argument("--config", type=str, help="Path to YAML config used during training.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file to evaluate.")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Dataset split to evaluate.")
    parser.add_argument("--device", type=str, help="Override compute device.")
    return parser.parse_args()


def evaluate_loader(model: torch.nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"Evaluate-{loader.dataset.__class__.__name__}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = override_config(config, {"device": args.device})

    device = get_device(config.device)
    _, val_loader, test_loader, class_names = create_dataloaders(config)
    loader = val_loader if args.split == "val" else test_loader

    model = build_model(num_classes=config.num_classes or len(class_names)).to(device)
    load_checkpoint(model, optimizer=None, path=args.checkpoint, device=device, scaler=None)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss, acc = evaluate_loader(model, loader, criterion, device)
    print(f"{args.split.capitalize()} loss: {loss:.4f} | {args.split.capitalize()} accuracy: {acc:.4%}")


if __name__ == "__main__":
    main()
