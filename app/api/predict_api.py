'''
API des endpoints de prédictions:
- post pour l'imputation
- get pour l'obtention de l'image relative à la prédiction
'''
# mpredict_api.py
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from utils.functions import predict_weather

router = APIRouter()
logger = logging.getLogger(__name__)

# Définition du chemin des images
BASE_PATH = Path(__file__).parent.parent
IMAGE_PATH = BASE_PATH / "static" / "images"

class WeatherData(BaseModel):
    '''
    Definition des features
    '''
    Location: str
    MinTemp: float
    MaxTemp: float
    WindGustDir: str
    WindGustSpeed: float
    WindDir9am: str
    WindDir3pm: str
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity9am: float
    Humidity3pm: float
    Pressure3pm: float
    Cloud9am: float
    Cloud3pm: float
    RainToday: str

@router.post("/predict")
async def predict(data: WeatherData):
    """
    Endpoint pour prédire le temps.
    """
    try:
        prediction, probability = predict_weather(data.model_dump())

        return {
            "status": "success",
            "prediction": "Yes" if prediction == 1 else "No",
            "probability": probability
        }
    except Exception as e:
        logger.error("Erreur lors de la prédiction: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.get("/image/{prediction}")
async def get_weather_image(prediction: str):
    '''
    Retourne l'image d'un soleil ou de pluie en fonction de la prediction
    '''
    try:
        image_path = IMAGE_PATH / f"{'sun' if prediction == 'Sun' else 'rain'}.jpeg"
        return FileResponse(image_path)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Image not found") from e
