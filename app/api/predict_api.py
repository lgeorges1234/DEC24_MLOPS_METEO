'''
API des endpoints de prédictions:
- post pour l'imputation
- get pour l'obtention de l'image relative à la prédiction
'''
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from utils.functions import predict_weather
import pandas as pd

router = APIRouter()
logger = logging.getLogger(__name__)

# Définition du chemin des images
BASE_PATH = Path(__file__).parent.parent
BASE_PATH_2 = Path(__file__).parent.parent.parent

IMAGE_PATH = BASE_PATH / "static" / "images"
CLEAN_DATA_PATH = Path("/app/api/data/prepared_data")
csv_daily_cleaned = "daily_row_prediction_cleaned.csv"


'''
@router.get("/predict")
async def predict(
    Location: int = Query(..., description="Location code"),
    MinTemp: float = Query(..., description="Minimum temperature"),
    MaxTemp: float = Query(..., description="Maximum temperature"),
    WindGustDir: float = Query(..., description="Wind gust direction"),
    WindGustSpeed: float = Query(..., description="Wind gust speed"),
    WindDir9am: float = Query(..., description="Wind direction at 9am"),
    WindDir3pm: float = Query(..., description="Wind direction at 3pm"),
    WindSpeed9am: float = Query(..., description="Wind speed at 9am"),
    WindSpeed3pm: float = Query(..., description="Wind speed at 3pm"),
    Humidity9am: float = Query(..., description="Humidity at 9am"),
    Humidity3pm: float = Query(..., description="Humidity at 3pm"),
    Pressure3pm: float = Query(..., description="Atmospheric pressure at 3pm"),
    Cloud9am: float = Query(..., description="Cloud cover at 9am"),
    Cloud3pm: float = Query(..., description="Cloud cover at 3pm"),
    RainToday: float = Query(..., description="Did it rain today"),
    RainTomorrow: float = Query(..., description="Will it rain tomorrow")
):
    try:
        # Créer un dictionnaire à partir des paramètres de requête
        data = {
            "Location": Location,
            "MinTemp": MinTemp,
            "MaxTemp": MaxTemp,
            "WindGustDir": WindGustDir,
            "WindGustSpeed": WindGustSpeed,
            "WindDir9am": WindDir9am,
            "WindDir3pm": WindDir3pm,
            "WindSpeed9am": WindSpeed9am,
            "WindSpeed3pm": WindSpeed3pm,
            "Humidity9am": Humidity9am,
            "Humidity3pm": Humidity3pm,
            "Pressure3pm": Pressure3pm,
            "Cloud9am": Cloud9am,
            "Cloud3pm": Cloud3pm,
            "RainToday": RainToday,
            "RainTomorrow": RainTomorrow
        }
        
        # Utiliser la fonction de prédiction existante
        prediction, probability = predict_weather(data)

        return {
            "status": "success",
            "prediction": "Good" if prediction == 1 else "Bad",
            "probability": probability
        }
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
'''

@router.get("/predict")
async def predict():
    '''
    Fonction de prediction
    Vérification de la présence des données journalières
    Prédiction en appelant la fonction predict_weather
    '''
    try:
        if not Path(CLEAN_DATA_PATH / csv_daily_cleaned).exists():
            raise HTTPException(status_code=404, detail="Aucun fichier journalier trouvé.")

        df = pd.read_csv(CLEAN_DATA_PATH / csv_daily_cleaned)

        if df.empty:
            raise HTTPException(status_code=400, detail="Le fichier CSV est vide.")

        prediction, probability = predict_weather()

        # Formater la réponse
        response =  {
                    "prediction": "Good" if prediction == 1 else "Bad",
                    "probability": probability
                    }


        return {"status": "success", "results": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
