'''
API des endpoints de prédictions:
- post pour l'imputation
- get pour l'obtention de l'image relative à la prédiction
'''
# model_api.py
from pathlib import Path
import pickle
import logging
import pandas as pd
from fastapi import HTTPException, APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel
import prepare_data  # Importation du script de préparation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

# Chemin vers le modèle
MODEL_PATH = Path('../models/rfc.pkl')
IMAGE_PATH = Path('../static/images')

class WeatherData(BaseModel):
    '''
    Features utilisés dans le fichier original
    Elles devront être saisies dans le formulaire de l'interface graphique
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
async def make_prediction(data: WeatherData):
    '''
    Fonction de prediction
    '''
    try:
        # Créer un DataFrame avec les données utilisateur
        input_df = pd.DataFrame([data.model_dump()])
        
        # Utiliser la fonction de préparation existante
        data_prepared = prepare_data(input_df)
        
        # Chargement du modèle et prédiction
        with open(MODEL_PATH, "rb") as rfc:
            model = pickle.load(rfc)
            
        prediction = model.predict(data_prepared)[0]
        probability = model.predict_proba(data_prepared)[0][1]
        
        return {
            "prediction": "Sun" if prediction == 1 else "Rain",
            "probability": float(probability),
            "image_url": f"/image/{'Sun' if prediction == 1 else 'Rain'}"
        }
        
    except Exception as e:
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