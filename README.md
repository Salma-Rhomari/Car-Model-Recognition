# Car Model Recognition 🚗

Upload a car photo and get a prediction of its **brand**, **model**, and (stretch goal) **approximate generation** — e.g. `BMW 3 Series (2019–2023)`.

Built with transfer learning on a pretrained CNN backbone, deployed as an interactive Streamlit demo.

## Demo

> Screenshot / GIF of the Streamlit app goes here once built.

## Project Goals

- Predict car **brand + model** from a single photo (e.g. BMW 3 Series, Mercedes C-Class, Audi A4)
- Stretch goal: predict approximate **generation/year range** as well
- Learn and demonstrate: CNN architectures, transfer learning, image preprocessing/augmentation, model evaluation, Grad-CAM interpretability

## Dataset

Using the [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) (196 classes, ~16,185 images, includes make/model/year annotations).

> Dataset is not committed to this repo — download instructions in `data/raw/README.md` (or see setup below).

## Project Structure

```
car-model-recognition/
├── data/               # raw & processed datasets (gitignored)
├── notebooks/          # EDA, preprocessing, baseline experiments
├── src/                # core source code (data loading, model, training, inference)
├── models/             # saved model weights (gitignored)
├── app/                # Streamlit demo app
├── tests/              # unit tests
└── outputs/            # sample predictions, training curves, plots
```

## Setup

```bash
git clone https://github.com/<your-username>/car-model-recognition.git
cd car-model-recognition
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

### Train the model
```bash
python -m src.train --epochs 20 --backbone resnet50
```

### Run inference on a single image
```bash
python -m src.predict --image path/to/car.jpg
```

### Launch the demo app
```bash
streamlit run app/streamlit_app.py
```

## Approach

1. **Transfer learning** on a pretrained backbone (ResNet50 / EfficientNet), fine-tuning the final layers on the car dataset.
2. **Data augmentation** (random crop, flip, color jitter) to improve generalization on a modest-sized dataset.
3. **Evaluation** via top-1/top-5 accuracy, confusion matrix, and per-class breakdown.
4. **Grad-CAM** visualizations to show which regions of the image the model uses to make predictions.

## Results

> To be filled in once training is complete: accuracy, confusion matrix, example predictions.

## Tech Stack

- PyTorch (model + training)
- Streamlit (demo app)
- Grad-CAM (interpretability)

## License

MIT
