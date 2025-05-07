from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: str
    latitude: float
    longitude: float