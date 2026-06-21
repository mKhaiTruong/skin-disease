from fastapi import FastAPI, UploadFile
from PIL import Image
import numpy as np
import io

import boto3
import os
from pathlib import Path

S3_BUCKET = os.getenv("S3_BUCKET")
S3_KEY    = os.getenv("S3_MODEL_KEY", "train/efficientnet/weights.onnx")
ONNX_PATH = Path("/app/artifacts/weights.onnx")

from vision.strategies.onnx_factory import ONNXModelFactory

NUM_CLASSES = 10
IMAGE_SIZE  = 224
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

DISEASE_CLASSES = [
    "Eczema",
    "Warts, Molluscum & Viral Infections",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma (BCC)",
    "Melanocytic Nevi (NV)",
    "Benign Keratosis-like Lesions (BKL)",
    "Psoriasis & Lichen Planus",
    "Seborrheic Keratoses & Benign Tumors",
    "Tinea, Ringworm & Fungal Infections",
]

app = FastAPI()

def preprocess(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)          # HWC -> CHW
    return np.expand_dims(arr, 0).astype(np.float32)  # add batch dim

def download_model():
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ONNX_PATH.exists():
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, S3_KEY, str(ONNX_PATH))
        
download_model()

model = ONNXModelFactory.create(model_name="efficientnet", num_classes=NUM_CLASSES)
model.load(ONNX_PATH)

@app.post("/predict")
async def predict(file: UploadFile):
    img_tensor = preprocess(await file.read())
    pred_idx, confidence = model.predict(img_tensor)
    return {"disease": DISEASE_CLASSES[pred_idx], "confidence": confidence}

@app.get("/health")
async def health():
    return {"status": "ok"}