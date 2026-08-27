"""
Run inference on a single image.

Usage:
    python -m src.predict --image path/to/car.jpg
"""

import argparse
import json

import torch
from PIL import Image

from src.data_loader import get_transforms
from src.model import build_model


def load_model(model_path: str, class_names_path: str, backbone: str, device):
    with open(class_names_path) as f:
        class_names = json.load(f)

    model = build_model(num_classes=len(class_names), backbone=backbone, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, class_names


@torch.no_grad()
def predict_image(model, class_names, image_path: str, device, top_k: int = 3):
    transform = get_transforms(train=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1).squeeze(0)
    top_probs, top_idxs = probs.topk(top_k)

    results = [
        {"class": class_names[idx], "probability": float(prob)}
        for prob, idx in zip(top_probs, top_idxs)
    ]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the car image")
    parser.add_argument("--model-path", default="models/best_model.pth")
    parser.add_argument("--class-names", default="models/class_names.json")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(args.model_path, args.class_names, args.backbone, device)
    results = predict_image(model, class_names, args.image, device, top_k=args.top_k)

    print(f"\nPredictions for {args.image}:")
    for r in results:
        print(f"  {r['class']:30s} {r['probability']*100:.2f}%")


if __name__ == "__main__":
    main()
