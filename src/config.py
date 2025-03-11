from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class TrainingConfig:
    """Serializable configuration container for training/eval scripts."""

    project_name: str = "cnn-image-classifier"
    dataset: str = "cifar10"
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "runs"
    seed: int = 42
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    label_smoothing: float = 0.0
    val_split: float = 0.1
    max_grad_norm: float = 2.0
    device: str = "cuda"
    use_amp: bool = True
    dataset_mean: Optional[list[float]] = None
    dataset_std: Optional[list[float]] = None
    num_classes: int = 10

    def ensure_dirs(self) -> None:
        """Create directories referenced by the config if they do not exist."""
        for folder in (self.data_dir, self.model_dir, self.log_dir):
            Path(folder).mkdir(parents=True, exist_ok=True)


def load_config(path: Optional[str]) -> TrainingConfig:
    """Load configuration values from YAML if provided, otherwise defaults."""
    config_dict: Dict[str, Any] = asdict(TrainingConfig())
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            loaded: Dict[str, Any] = yaml.safe_load(handle) or {}
            config_dict.update(loaded)
    return TrainingConfig(**config_dict)


def save_config(config: TrainingConfig, path: str | Path) -> None:
    """Persist the configuration to disk as YAML."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)


def override_config(config: TrainingConfig, overrides: Dict[str, Any]) -> TrainingConfig:
    """Create a new config using CLI overrides."""
    base = asdict(config)
    for key, value in overrides.items():
        if value is not None and key in base:
            base[key] = value
    return TrainingConfig(**base)
