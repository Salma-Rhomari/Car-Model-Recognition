# Car Model Recognition 

Upload a car photo and get a prediction of its **brand**, **model**, and (stretch goal) **approximate generation** — e.g. `BMW 3 Series (2019–2023)`.

Built with transfer learning on a pretrained CNN backbone, deployed as an interactive Streamlit demo.

## Project Goals

- Predict car **brand + model** from a single photo (e.g. BMW 3 Series, Mercedes C-Class, Audi A4)
- Stretch goal: predict approximate **generation/year range** as well
- Learn and demonstrate: CNN architectures, transfer learning, image preprocessing/augmentation, model evaluation, Grad-CAM interpretability
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

## Tech Stack

- PyTorch (model + training)
- Streamlit (demo app)
- Grad-CAM (interpretability)

## License

MIT
