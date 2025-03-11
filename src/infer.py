from __future__ import annotations

import argparse

import torch
from PIL import Image

from .config import load_config, override_config
from .data import build_transforms
from .model import build_model
from .utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--config", type=str, help="Config file used during training.")
    parser.add_argument("--device", type=str, help="Override device (cuda/mps/cpu).")
    return parser.parse_args()


def load_image(path: str, transform) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = override_config(config, {"device": args.device})

    device = get_device(config.device)
    _, eval_transform = build_transforms(config)

    model = build_model(num_classes=config.num_classes).to(device)
    load_checkpoint(model, optimizer=None, path=args.checkpoint, device=device, scaler=None)
    model.eval()

    tensor = load_image(args.image, eval_transform).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)

    class_id = probabilities.argmax().item()
    confidence = probabilities[class_id].item()
    print(f"Predicted class index: {class_id} with confidence {confidence:.4f}")


if __name__ == "__main__":
    main()
