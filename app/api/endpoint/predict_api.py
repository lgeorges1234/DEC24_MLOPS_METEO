# predict_api.py
'''
API de predection
Elle utilise le modele present dans le modele registry de MLflow
pour transmettre une prediction soit automatiquement en utilisant la ligne de prediction du jour - methode GET -
soit a traver un fichier csv fournit en POST request.
'''
from datetime import datetime
from http.client import HTTPException
import logging
import mlflow
import pandas as pd
from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any
from pydantic import BaseModel, ValidationError, Field, validator
from utils.mlflow_config import setup_mlflow
from utils.mlflow_run_manager import get_deployment_run
from utils.functions import PREDICTION_RAW_DATA_PATH, predict_weather, csv_file_daily_prediction

router = APIRouter()
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    """Modèle pour les demandes de prédiction manuelles"""
    # location: str
    date: str
    features: Dict[str, Any]
    run_id: Optional[str] = None


@router.get("/predict")
async def automatic_predict():
    """
    Endpoint for inference that relies solely on MLflow for model retrieval.
    Uses the same data feeding approach as the manual prediction endpoint.
    """
    try:
        # Initialize response variables
        status = "success"
        message = "Daily prediction successfully completed"
        
        # Set up MLflow with both tracking and registry URIs
        try:
            setup_mlflow()
            logger.info(f"MLflow connection check successful")
        except Exception as e:
            error_msg = f"Failed to connect to MLflow server: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=503, detail=error_msg)
            
        # Verify daily prediction file exists
        prediction_file = PREDICTION_RAW_DATA_PATH / csv_file_daily_prediction
        if not prediction_file.exists():
            raise FileNotFoundError(f"Prediction file not found: {prediction_file}")
            
        # Read the file content and convert to dictionary (similar to user input)
        try:
            df = pd.read_csv(prediction_file)
            if df.empty:
                raise ValueError("Daily prediction file is empty")
                
            # Remove the target column if it exists (to match manual prediction behavior)
            if 'RainTomorrow' in df.columns:
                logger.info("Removing RainTomorrow target column from input data")
                df = df.drop(columns=['RainTomorrow'])
            
            # Extract the first row as a dictionary to use as input
            row_data = df.iloc[0].to_dict()
            logger.info(f"Daily prediction row extracted: {row_data}")
            
            # Create a hash of the data for tracking changes
            data_hash = hash(str(row_data))
            logger.info(f"Daily prediction data hash: {data_hash}")
        except Exception as e:
            logger.error(f"Error reading prediction file: {str(e)}")
            raise
            
        # Get deployment run for the current model
        try:
            deployment_run_id, model_version = get_deployment_run()
            logger.info(f"Using model version {model_version} with deployment run {deployment_run_id}")
        except Exception as e:
            error_msg = f"Failed to get deployment run: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Create a nested run for this specific prediction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_run_name = f"automatic_prediction_V{model_version}_{timestamp}"
        
        # Use the deployment run as parent
        with mlflow.start_run(run_id=deployment_run_id):
            with mlflow.start_run(run_name=prediction_run_name, nested=True) as nested_run:
                # Add tags to identify this run
                mlflow.set_tag("pipeline_type", "prediction")
                mlflow.set_tag("prediction_date", timestamp)
                mlflow.set_tag("endpoint", "predict_api")
                mlflow.set_tag("model_version", str(model_version))
                mlflow.set_tag("data_hash", str(data_hash))
                
                # Log input data
                for key, value in row_data.items():
                    try:
                        # Convert to appropriate type and log as parameter
                        if pd.isna(value):
                            param_value = "NA"
                        else:
                            param_value = str(value)
                        mlflow.log_param(key, param_value)
                    except Exception as e:
                        logger.warning(f"Could not log parameter {key}: {str(e)}")
                
                # Execute the prediction with the row data as input (just like user input)
                prediction, probability = predict_weather(user_input=row_data)
                
                return {
                    "status": status,
                    "message": message,
                    "run_id": nested_run.info.run_id,
                    "deployment_run_id": deployment_run_id,
                    "model_version": model_version,
                    "prediction": prediction,
                    "probability": probability,
                    "input_data_hash": str(data_hash)
                }
                
    except Exception as e:
        # Clean up any active runs
        if mlflow.active_run():
            mlflow.end_run()
            
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
class UserInputPrediction(BaseModel):
    Location: int = Field(
        ..., 
        gt=0, 
        description="Location code (positive integer)"
    )
    MinTemp: float = Field(
        ..., 
        ge=-50, 
        le=60, 
        description="Minimum temperature between -50 and 60 degrees"
    )
    MaxTemp: float = Field(
        ..., 
        ge=-50, 
        le=60, 
        description="Maximum temperature between -50 and 60 degrees"
    )
    WindGustDir: float = Field(
        ..., 
        ge=0, 
        le=360, 
        description="Wind gust direction in degrees (0-360)"
    )
    WindGustSpeed: float = Field(
        ..., 
        ge=0, 
        le=200, 
        description="Wind gust speed (0-200)"
    )
    WindDir9am: float = Field(
        ..., 
        ge=0, 
        le=360, 
        description="Wind direction at 9am in degrees (0-360)"
    )
    WindDir3pm: float = Field(
        ..., 
        ge=0, 
        le=360, 
        description="Wind direction at 3pm in degrees (0-360)"
    )
    WindSpeed9am: float = Field(
        ..., 
        ge=0, 
        le=200, 
        description="Wind speed at 9am (0-200)"
    )
    WindSpeed3pm: float = Field(
        ..., 
        ge=0, 
        le=200, 
        description="Wind speed at 3pm (0-200)"
    )
    Humidity9am: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Humidity at 9am (0-100%)"
    )
    Humidity3pm: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Humidity at 3pm (0-100%)"
    )
    Pressure3pm: float = Field(
        ..., 
        ge=900, 
        le=1100, 
        description="Atmospheric pressure at 3pm (900-1100 hPa)"
    )
    Cloud9am: float = Field(
        ..., 
        ge=0, 
        le=9, 
        description="Cloud cover at 9am (0-9)"
    )
    Cloud3pm: float = Field(
        ..., 
        ge=0, 
        le=9, 
        description="Cloud cover at 3pm (0-9)"
    )
    RainToday: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Did it rain today (0 or 1)"
    )

    @validator('MaxTemp')
    def check_max_temp_greater_than_min(cls, v, values):
        """Ensure MaxTemp is not less than MinTemp"""
        if 'MinTemp' in values and v < values['MinTemp']:
            raise ValueError("MaxTemp must be greater than or equal to MinTemp")
        return v

    @validator('WindDir9am', 'WindDir3pm', 'WindGustDir')
    def validate_wind_direction(cls, v):
        """Ensure wind direction is between 0 and 360 degrees"""
        if not (0 <= v <= 360):
            raise ValueError("Wind direction must be between 0 and 360 degrees")
        return v

    @validator('RainToday')
    def validate_rain_binary(cls, v):
        """Ensure rain values are binary"""
        if v not in [0, 1]:
            raise ValueError("Rain values must be 0 or 1")
        return v
    
    @validator('Cloud9am', 'Cloud3pm')
    def validate_cloud_coverage(cls, v):
        """Ensure cloud coverage"""
        if not (0 <= v <= 9):
            raise ValueError("Cloud coverage must be between 0 and 9")
        return v
    
    @validator('Humidity9am', 'Humidity3pm')
    def validate_humidity_level(cls, v):
        """Ensure humidity level"""
        if not (0 <= v <= 100):
            raise ValueError("Humidity must be between 0 and 100%")
        return v
    
    @validator('Pressure3pm')
    def validate_pressure_level(cls, v):
        """Ensure pressure"""
        if not (900 <= v <= 1100):
            raise ValueError("Pressure must be between 900 and 1100 hPa")
        return v
    


    class Config:
        # Permet de lever une erreur si des champs supplémentaires sont envoyés
        extra = 'forbid'
    
@router.post("/predict_user")
async def predict_user_input(input_data: UserInputPrediction):
    """
    Endpoint for manual user input inference.
    Creates a new prediction run under the current model deployment run.
    """
    try:
        # Initialize response variables
        status = "success"
        message = "Manual prediction successfully completed"
        
        # Get deployment run for the current model
        # This function already calls setup_mlflow() internally
        deployment_run_id, model_version = get_deployment_run()
        
        # Create a nested run for this specific prediction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_run_name = f"manual_prediction_V{model_version}_{timestamp}"
        
        # Convert Pydantic model to dictionary
        data = input_data.model_dump()

        # Log the exact data received
        logger.info(f"Received prediction data: {data}")
        
        # Use the deployment run as parent
        with mlflow.start_run(run_id=deployment_run_id):
            with mlflow.start_run(run_name=prediction_run_name, nested=True) as nested_run:
                # Add tags to identify this run
                mlflow.set_tag("pipeline_type", "manual_prediction")
                mlflow.set_tag("prediction_date", timestamp)
                mlflow.set_tag("endpoint", "predict_user_api")
                mlflow.set_tag("model_version", str(model_version))
                
                # Log input data
                mlflow.log_params(data)
                
                # Execute the prediction with user input
                try:
                    prediction, probability = predict_weather(user_input=data)
                except Exception as predict_error:
                    logger.error(f"Prediction error: {str(predict_error)}")
                    raise
                
                return {
                    "status": status,
                    "message": message,
                    "run_id": nested_run.info.run_id,
                    "deployment_run_id": deployment_run_id,
                    "model_version": model_version,
                    "prediction": prediction,
                    "probability": probability
                }
                
    except ValidationError as ve:
        # Gestion des erreurs de validation
        raise HTTPException(
            status_code=422, 
            detail=[{"loc": e["loc"], "msg": e["msg"]} for e in ve.errors()]
        )
    except Exception as e:
        # Clean up any active runs
        if mlflow.active_run():
            mlflow.end_run()
            
        logger.error(f"Error during manual prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))