'''
API des endpoints de prédictions:
- post pour l'imputation
- get pour l'obtention de l'image relative à la prédiction
'''
# mpredict_api.py
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator
from utils.functions import predict_weather

router = APIRouter()
logger = logging.getLogger(__name__)

# Définition du chemin des images
BASE_PATH = Path(__file__).parent.parent
IMAGE_PATH = BASE_PATH / "static" / "images"

# Pydantic model for data validation
class WeatherData(BaseModel):
    Location: int = Field(..., description="Location code")
    MinTemp: float = Field(..., description="Minimum temperature")
    MaxTemp: float = Field(..., description="Maximum temperature")
    WindGustDir: float = Field(..., description="Wind gust direction")
    WindGustSpeed: float = Field(..., ge=0, description="Wind gust speed")
    WindDir9am: float = Field(..., description="Wind direction at 9am")
    WindDir3pm: float = Field(..., description="Wind direction at 3pm")
    WindSpeed9am: float = Field(..., ge=0, description="Wind speed at 9am")
    WindSpeed3pm: float = Field(..., ge=0, description="Wind speed at 3pm")
    Humidity9am: float = Field(..., ge=0, le=100, description="Humidity at 9am")
    Humidity3pm: float = Field(..., ge=0, le=100, description="Humidity at 3pm")
    Pressure3pm: float = Field(..., description="Atmospheric pressure at 3pm")
    Cloud9am: float = Field(..., ge=0, le=9, description="Cloud cover at 9am")
    Cloud3pm: float = Field(..., ge=0, le=9, description="Cloud cover at 3pm")
    RainToday: float = Field(..., ge=0, le=1, description="Did it rain today")
    RainTomorrow: float = Field(..., ge=0, le=1, description="Will it rain tomorrow")

    # Validator to check consistency between MinTemp and MaxTemp
    @field_validator('MaxTemp')
    def check_temperature(self, v):
        if v < self.MinTemp:
            raise ValueError("Maximum temperature must be greater than or equal to minimum temperature")
        return v

# Function to validate query parameters
def validate_weather_data(
    Location: int,
    MinTemp: float,
    MaxTemp: float,
    WindGustDir: float,
    WindGustSpeed: float,
    WindDir9am: float,
    WindDir3pm: float,
    WindSpeed9am: float,
    WindSpeed3pm: float,
    Humidity9am: float,
    Humidity3pm: float,
    Pressure3pm: float,
    Cloud9am: float,
    Cloud3pm: float,
    RainToday: float,
    RainTomorrow: float
) -> WeatherData:
    """Validate the query parameters against the WeatherData model"""
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
    return WeatherData(**data)

@router.get("/predict")
async def predict(weather_data: WeatherData = Depends(validate_weather_data)):
    try:
        # Using the validated Pydantic model
        data = weather_data.model_dump()
        
        prediction, probability = predict_weather(data)

        return {
            "status": "success",
            "prediction": "Good" if prediction == 1 else "Bad",
            "probability": probability
        }
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e