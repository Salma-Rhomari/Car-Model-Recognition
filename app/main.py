import base64
import io
import json

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data_loader import get_transforms
from src.model import build_model
from src.gradcam import get_target_layer

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

target_layers = get_target_layer(model, BACKBONE)
cam = GradCAM(model=model, target_layers=target_layers)

app = FastAPI(title="Car Model Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
        top_probs, top_indices = torch.topk(probs, k=3)

    predictions = [
        {"class_name": class_names[idx.item()], "confidence": round(prob.item(), 4)}
        for prob, idx in zip(top_probs, top_indices)
    ]

    rgb_img = np.array(pil_image.resize((224, 224))) / 255.0
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    visualization = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True)

    gradcam_pil = Image.fromarray(visualization)
    buffer = io.BytesIO()
    gradcam_pil.save(buffer, format="PNG")
    gradcam_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "predicted_class": predictions[0]["class_name"],
        "confidence": predictions[0]["confidence"],
        "top_predictions": predictions,
        "gradcam_image": f"data:image/png;base64,{gradcam_base64}",
    }