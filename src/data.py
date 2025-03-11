from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import TrainingConfig

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms(config: TrainingConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train/eval transforms for the configured dataset."""
    dataset = config.dataset.lower()
    if dataset != "cifar10":
        raise ValueError(f"Unsupported dataset '{config.dataset}'. Only 'cifar10' is built-in.")

    mean = config.dataset_mean or CIFAR10_MEAN
    std = config.dataset_std or CIFAR10_STD

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, eval_transform


def create_dataloaders(config: TrainingConfig):
    """Create train/val/test dataloaders and the list of class names."""
    config.ensure_dirs()
    train_transform, eval_transform = build_transforms(config)

    dataset_root = Path(config.data_dir)
    dataset_root.mkdir(parents=True, exist_ok=True)

    base_train = datasets.CIFAR10(
        root=dataset_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    base_val = datasets.CIFAR10(
        root=dataset_root,
        train=True,
        download=False,
        transform=eval_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=dataset_root,
        train=False,
        download=True,
        transform=eval_transform,
    )

    generator = torch.Generator().manual_seed(config.seed)
    val_size = max(1, int(len(base_train) * config.val_split))
    train_size = len(base_train) - val_size
    indices = torch.randperm(len(base_train), generator=generator)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = Subset(base_train, train_indices)
    val_subset = Subset(base_val, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader, test_loader, base_train.classes
