from fastapi import FastAPI, UploadFile, File
from app.model import predict_pest
from app.schema import PredictionResponse
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
