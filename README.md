# Car Model Recognition 

Upload a car photo and get a prediction of its **brand**, **model**, and (stretch goal) **approximate generation** — e.g. `BMW 3 Series (2019–2023)`.
## Project Goals

- Predict car brand + model + year from a single photo (e.g. BMW 3 Series 2012, Mercedes C-Class 2015)
- Learn and demonstrate: CNN architectures, transfer learning, image preprocessing/augmentation, model evaluation, Grad-CAM interpretability
- Serve the model through a real API and a usable web interface
- 
 ## Tech Stack

- **Model:** PyTorch, torchvision (ResNet50, transfer learning)
- **Interpretability:** Grad-CAM (`pytorch-grad-cam`), generated dynamically per uploaded image
- **Backend API:** FastAPI + Uvicorn
- **Frontend:** Next.js + Tailwind CSS
- 
## Setup

```bash
git clone https://github.com/<your-username>/car-model-recognition.git
cd car-model-recognition
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Approach

1. **Transfer learning** on a pretrained backbone (ResNet50 / EfficientNet), fine-tuning the final layers on the car dataset.
2. **Data augmentation** (random crop, flip, color jitter) to improve generalization on a modest-sized dataset.
3. **Evaluation** via top-1/top-5 accuracy, confusion matrix, and per-class breakdown.
4. **Grad-CAM** visualizations to show which regions of the image the model uses to make predictions.

## Results

Final model: **ResNet50** (transfer learning, fine-tuned on last layers)

| Stage | Val Accuracy |
|---|---|
| Frozen backbone (baseline) | 15.19% → 40.76% |
| Fine-tuning (layer4 + fc) | 67.34% → 69.45% |
| + stronger augmentation, dropout, lower LR, weight decay | **72.74%** |

- Target (top-1 accuracy ≥ 70%): **achieved**
- Backbone: ResNet50 pretrained on ImageNet
- Fine-tuned `layer4` + `fc` layers, with dropout (0.3), data augmentation (crop, flip, rotation, color jitter), Adam (lr=3e-5, weight_decay=1e-4), `ReduceLROnPlateau` scheduler
- Saved model: `resnet50_best.pth`

## License

MIT
