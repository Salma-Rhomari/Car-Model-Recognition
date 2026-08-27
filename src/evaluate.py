"""
Evaluation utilities: confusion matrix, per-class accuracy, top-k accuracy.

Usage:
    python -m src.evaluate --model-path models/best_model.pth
"""

import argparse
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from src.data_loader import get_dataloaders
from src.model import build_model


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, class_names, output_path="outputs/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, cmap="Blues", cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved confusion matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model-path", default="models/best_model.pth")
    parser.add_argument("--backbone", default="resnet50")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("models/class_names.json") as f:
        class_names = json.load(f)

    _, _, test_loader, _ = get_dataloaders(args.data_dir)

    model = build_model(num_classes=len(class_names), backbone=args.backbone, freeze_backbone=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    y_pred, y_true = get_predictions(model, test_loader, device)

    print(classification_report(y_true, y_pred, target_names=class_names))
    plot_confusion_matrix(y_true, y_pred, class_names)


if __name__ == "__main__":
    main()
