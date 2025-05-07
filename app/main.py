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
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Log the size of the image for debugging
        print(f"Received image with size: {image.size}")
        
        prediction = predict_pest(image)
        
        # Log the prediction result
        print(f"Prediction: {prediction}")
        
        return PredictionResponse(prediction=prediction, latitude=latitude, longitude=longitude)
    except Exception as e:
        # Log any exception during processing
        print(f"Error during prediction: {e}")
        return {"error": "Prediction failed."}

@app.get("/")
def read_root():
    return {
        "message": "Onion Scan API is running!",
        "endpoints": {
            "POST /predict": "Send an image with latitude and longitude to get pest prediction."
        }
    }
