from fastapi import FastAPI, UploadFile
from PIL import Image
import io
import torch
from torchvision import transforms
 
from vision.strategies.onnx_factory import ONNXModelFactory
NUM_CLASSES = 10
MODEL_NAME  = "efficientnet"
ONNX_PATH   = "artifacts/train/efficientnet/weights.onnx"
IMAGE_SIZE  = 224

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

app = FastAPI(title="Vision Service")

model = ONNXModelFactory.create(model_name=MODEL_NAME, num_classes=NUM_CLASSES)
model.load(ONNX_PATH)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
 
    pred_idx, confidence = model.predict(img_tensor)
    disease = DISEASE_CLASSES[pred_idx]
 
    return {"disease": disease, "confidence": confidence}
 
 
@app.get("/health")
async def health():
    return {"status": "ok"}