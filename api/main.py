import os
import sys
import io
import base64

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

import torch

# 🔁 Relative imports for utils inside api/
from .model_utils import load_model, preprocess_image
from .face_utils import detect_and_crop_face
print("✅ FastAPI starting")

# 🚀 FastAPI app init
app = FastAPI()

# 🌐 CORS setup - allow all origins for now (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# ✅ Load model and device at startup
model, device = load_model()

# 📷 Request schema for base64-encoded image
class ImagePayload(BaseModel):
    image_base64: str

# 🎯 POST endpoint for age prediction
@app.post("/predict")
def predict_base64(payload: ImagePayload):
    try:
        # Decode and convert to PIL
        image_data = base64.b64decode(payload.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Detect face and crop
        face_image = detect_and_crop_face(image)

        # Preprocess and predict
        image_tensor = preprocess_image(face_image).to(device)
        with torch.no_grad():
            age_pred = model(image_tensor)
            predicted_age = int(age_pred.item())

        return {"predicted_age": predicted_age}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🧠 Render requires dynamic port binding
#if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render assigns this automatically
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)
