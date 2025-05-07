from fastapi import FastAPI, UploadFile, File
from app.model import predict_pest
from app.schema import PredictionResponse  # Make sure this file exists and is correct
from PIL import Image
import io

app = FastAPI()

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    latitude: float = 0.0,
    longitude: float = 0.0
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    prediction = predict_pest(image)
    return PredictionResponse(prediction=prediction, latitude=latitude, longitude=longitude)

@app.get("/")
def read_root():
    return {
        "message": "Onion Scan API is running!",
        "endpoints": {
            "POST /predict": "Send an image with latitude and longitude to get pest prediction."
        }
    }
