from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import base64
import os
import requests

# =========================
# MODEL DEFINITION (RegNetMTL)
# =========================
class RegNetMTL(nn.Module):
    def __init__(self):
        super().__init__()
        regnet = models.regnet_y_400mf(
            weights=models.RegNet_Y_400MF_Weights.DEFAULT
        )

        self.stem = regnet.stem
        self.trunk = regnet.trunk_output  # (B, 440, 7, 7)

        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(440, 1)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            block(440, 256),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            block(256, 128),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            block(128, 64),
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            block(64, 32),
        )
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        feat = self.trunk(x)

        pooled = self.cls_pool(feat).flatten(1)
        cls_logit = self.cls_head(pooled).squeeze(1)

        y = self.up1(feat)
        y = self.up2(y)
        y = self.up3(y)
        y = self.up4(y)
        y = self.up5(y)
        seg_logit = self.seg_head(y).squeeze(1)

        return cls_logit, seg_logit


# =========================
# FASTAPI SETUP
# =========================
app = FastAPI(title="SkinGuard AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GLOBALS
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "backend/model"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

MODEL_URL = (
    "https://huggingface.co/ashmikotian7/Skin_Cancer/resolve/main/best_model.pth"
)

model = None


# =========================
# DOWNLOAD MODEL (HF)
# =========================
def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists")
        return

    print("⬇️ Downloading model from Hugging Face...")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("✅ Model downloaded successfully")


# =========================
# LOAD MODEL ON STARTUP
# =========================
@app.on_event("startup")
def load_model():
    global model
    print("🔄 Starting model initialization...")

    download_model()

    model = RegNetMTL().to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("🚀 Model loaded and ready")


# =========================
# IMAGE PREPROCESSING
# =========================
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


# =========================
# HEATMAP FUNCTION
# =========================
def apply_heatmap(prob_mask):
    h, w = prob_mask.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)

    mask_scaled = (prob_mask * 255).astype(np.uint8)
    heatmap[..., 0] = mask_scaled
    heatmap[..., 1] = np.clip(mask_scaled - 128, 0, 255)
    heatmap[..., 2] = np.clip(mask_scaled - 192, 0, 255)

    return heatmap


# =========================
# PREDICTION ROUTE
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if model is None:
            return JSONResponse(
                {"error": "Model not loaded yet"}, status_code=503
            )

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_size = image.size

        input_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            cls_logit, seg_logit = model(input_tensor)

            prob = torch.sigmoid(cls_logit).item()
            prediction = "Cancerous" if prob >= 0.5 else "Non-Cancerous"

            mask_prob = torch.sigmoid(seg_logit).squeeze().cpu().numpy()

            mask_resized = np.array(
                Image.fromarray(mask_prob).resize(
                    orig_size[::-1], resample=Image.BILINEAR
                )
            )

            mask_color = apply_heatmap(
                mask_resized / (mask_resized.max() + 1e-6)
            )

            mask_img = Image.fromarray(mask_color)
            buffer = io.BytesIO()
            mask_img.save(buffer, format="PNG")

            mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "predicted_class": prediction,
            "confidence": prob,
            "mask_available": True,
            "mask_image": f"data:image/png;base64,{mask_base64}",
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def root():
    return {"message": "SkinGuard AI backend is running 🚀"}
