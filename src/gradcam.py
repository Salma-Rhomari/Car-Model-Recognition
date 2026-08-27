"""
Grad-CAM visualization: shows which regions of the image the model
focused on to make its prediction. Great for the demo/portfolio appeal.

Usage:
    python -m src.gradcam --image path/to/car.jpg

Requires: pip install grad-cam
"""

import argparse
import json

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data_loader import get_transforms
from src.model import build_model


def get_target_layer(model, backbone: str):
    """Return the last convolutional layer to hook Grad-CAM onto."""
    if backbone == "resnet50":
        return [model.layer4[-1]]
    elif backbone == "efficientnet_b0":
        return [model.features[-1]]
    raise ValueError(f"Unsupported backbone: {backbone}")


def generate_gradcam(model, class_names, image_path: str, backbone: str, device, output_path="outputs/gradcam_result.png"):
    transform = get_transforms(train=False)
    pil_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Normalized image for overlay (0-1 range, resized)
    rgb_img = np.array(pil_image.resize((224, 224))) / 255.0

    target_layers = get_target_layer(model, backbone)
    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True)

    Image.fromarray(visualization).save(output_path)
    print(f"Saved Grad-CAM visualization to {output_path}")

    # Also print the prediction for context
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_idx = outputs.argmax(dim=1).item()
    print(f"Predicted class: {class_names[pred_idx]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-path", default="models/best_model.pth")
    parser.add_argument("--class-names", default="models/class_names.json")
    parser.add_argument("--backbone", default="resnet50")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.class_names) as f:
        class_names = json.load(f)

    model = build_model(num_classes=len(class_names), backbone=args.backbone, freeze_backbone=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    generate_gradcam(model, class_names, args.image, args.backbone, device)


if __name__ == "__main__":
    main()
