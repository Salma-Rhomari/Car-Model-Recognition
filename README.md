# Car Model Recognition 

Upload a car photo and get a prediction of its **brand**, **model**, and (stretch goal) **approximate generation** — e.g. `BMW 3 Series (2019–2023)`.
## Project Goals

- Predict car brand + model + year from a single photo (e.g. BMW 3 Series 2012, Mercedes C-Class 2015)
- Learn and demonstrate: CNN architectures, transfer learning, image preprocessing/augmentation, model evaluation, Grad-CAM interpretability
- Serve the model through a real API and a usable web interface
## Live Demo

A full-stack app: FastAPI backend serving the trained model, Next.js frontend for uploading photos and viewing predictions with Grad-CAM overlays.

 ## Tech Stack

- **Model:** PyTorch, torchvision (ResNet50, transfer learning)
- **Interpretability:** Grad-CAM (`pytorch-grad-cam`), generated dynamically per uploaded image
- **Backend API:** FastAPI + Uvicorn
- **Frontend:** Next.js + Tailwind CSS
- 

## Approach

- Transfer learning on ResNet50 (pretrained on ImageNet), fine-tuning the final layers on the Stanford Cars Dataset (196 classes, 8,144 images).
- Data augmentation (random crop, flip, rotation, color jitter) to improve generalization on a modest-sized dataset.
- Evaluation via test accuracy and confusion matrix.
- Grad-CAM visualizations, generated live on each prediction, to show which regions of the image the model uses to make its decision.

## Results

Final model: ResNet50 (transfer learning, fine-tuned on last layers)

| Stage | Val Accuracy |
|---|---|
| Frozen backbone (baseline) | 15.19% → 40.76% |
| Fine-tuning (layer4 + fc) | 67.34% → 69.45% |
| + stronger augmentation, dropout, lower LR, weight decay | 72.74% |
| **Final test accuracy** | **74%** |

Target (top-1 accuracy ≥ 70%): achieved.

- Backbone: ResNet50 pretrained on ImageNet
- Fine-tuned `layer4` + `fc` layers, with dropout (0.3), data augmentation (crop, flip, rotation, color jitter), Adam (lr=3e-5, weight_decay=1e-4), ReduceLROnPlateau scheduler
- Saved model: `resnet50_best.pth`

## License

MIT
