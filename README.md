# CNN Image Classifier (PyTorch)

Compact, reproducible PyTorch project for training, evaluating, and deploying a convolutional neural network on CIFAR-10 or other CIFAR-like datasets. The repository comes with opinionated defaults, AMP training, YAML-based hyperparameter management, and separate scripts for evaluation and single-image inference.

## Features
- PyTorch training loop with validation split, learning-rate scheduling, AMP, and checkpointing
- Modular `src/` package that separates configuration, data pipelines, models, and utilities
- Configurable via CLI flags or YAML files under `configs/`
- Ready-to-run CIFAR-10 setup that automatically downloads the dataset
- Evaluation and inference scripts for quick experimentation or deployment demos

## Project Structure
```
CNN-Image-Classifier-PyTorch
+-- configs/             # YAML configs (cifar10.yaml provided)
+-- requirements.txt
+-- README.md
+-- src/
    +-- config.py        # Dataclass config loader & overrides
    +-- data.py          # Dataset transforms + dataloader factory
    +-- model.py         # Simple CNN architecture
    +-- train.py         # Training entrypoint
    +-- evaluate.py      # Validation/Test evaluation
    +-- infer.py         # Single-image inference helper
    +-- utils.py         # Reusable helpers (seeding, checkpoints, etc.)
```

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate           # PowerShell on Windows
pip install -r requirements.txt
```

## Training
Use the default CIFAR-10 configuration (downloads dataset on first run):
```bash
python -m src.train --config configs/cifar10.yaml
```

Common overrides:
```bash
python -m src.train --config configs/cifar10.yaml --epochs 40 --batch-size 256 --learning-rate 5e-4
python -m src.train --config configs/cifar10.yaml --device cpu --no-amp
```

Checkpoints are stored under the `model_dir` defined in your config (default `artifacts/`). Metrics are appended to `log_dir/metrics.jsonl`.

## Evaluation
Evaluate the best checkpoint on the validation or test split:
```bash
python -m src.evaluate --config configs/cifar10.yaml --checkpoint artifacts/cifar10-cnn.pt --split test
```

## Single-Image Inference
```bash
python -m src.infer --config configs/cifar10.yaml --checkpoint artifacts/cifar10-cnn.pt --image path/to/image.png
```

The script prints the predicted class index and confidence. Extend `infer.py` with the class names from your dataset if you need human-readable labels.

## Custom Datasets
This starter focuses on CIFAR-10. To plug in a different dataset:
1. Add your transforms and dataset loading logic to `src/data.py`.
2. Update `TrainingConfig` (classes, normalization stats, etc.).
3. Point the config YAML to your data directory and set `dataset` to an identifier you handle in `data.py`.

## Next Steps
- Experiment with deeper architectures in `src/model.py`.
- Add experiment tracking integrations (Weights & Biases, MLflow).
- Extend the dataloader factory with a `ImageFolder` path for custom datasets.
