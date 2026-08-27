"""
Streamlit demo: upload a car photo, get brand/model predictions.

Run with:
    streamlit run app/streamlit_app.py
"""

import json
import os
import sys

import streamlit as st
import torch
from PIL import Image

# Allow importing from src/ when run as `streamlit run app/streamlit_app.py`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import get_transforms
from src.model import build_model

MODEL_PATH = "models/best_model.pth"
CLASS_NAMES_PATH = "models/class_names.json"
BACKBONE = "resnet50"


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)

    model = build_model(num_classes=len(class_names), backbone=BACKBONE, freeze_backbone=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, class_names, device


def predict(model, class_names, device, image: Image.Image, top_k: int = 3):
    transform = get_transforms(train=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        top_probs, top_idxs = probs.topk(top_k)

    return [
        {"class": class_names[idx], "probability": float(prob)}
        for prob, idx in zip(top_probs, top_idxs)
    ]


def main():
    st.set_page_config(page_title="Car Model Recognition", page_icon="🚗", layout="centered")
    st.title("🚗 Car Model Recognition")
    st.write("Upload a photo of a car and I'll predict the brand and model.")

    if not os.path.exists(MODEL_PATH):
        st.warning(
            "No trained model found yet. Train one first with:\n\n"
            "`python -m src.train --epochs 20`"
        )
        return

    uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)

        with st.spinner("Analyzing..."):
            model, class_names, device = load_model()
            results = predict(model, class_names, device, image)

        st.subheader("Predictions")
        for r in results:
            label = r["class"].replace("_", " ").title()
            st.write(f"**{label}** — {r['probability']*100:.1f}%")
            st.progress(r["probability"])


if __name__ == "__main__":
    main()
