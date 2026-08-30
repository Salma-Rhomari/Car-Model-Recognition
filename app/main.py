import io
import json

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.data_loader import get_transforms
from src.model import build_model

MODEL_PATH = "models/resnet50_best.pth"
CLASS_NAMES_PATH = "models/class_names.json"
BACKBONE = "resnet50"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)

model = build_model(num_classes=len(class_names), backbone=BACKBONE, freeze_backbone=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = get_transforms(train=False)

app = FastAPI(title="Car Model Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "num_classes": len(class_names)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)

    return {
        "predicted_class": class_names[top_idx.item()],
        "confidence": round(top_prob.item(), 4),
    }