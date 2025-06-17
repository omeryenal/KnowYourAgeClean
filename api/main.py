from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import base64, io

from .model_utils import load_model, preprocess_image
from .face_utils import detect_and_crop_face

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, device = load_model()

class ImagePayload(BaseModel):
    image_base64: str

@app.post("/predict")
def predict_base64(payload: ImagePayload):
    try:
        image_data = base64.b64decode(payload.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        face_image = detect_and_crop_face(image)
        image_tensor = preprocess_image(face_image).to(device)
        with torch.no_grad():
            age_pred = model(image_tensor)
            predicted_age = int(age_pred.item())
        return {"predicted_age": predicted_age}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
