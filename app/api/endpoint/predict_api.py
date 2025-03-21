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
from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any
from pydantic import BaseModel
from utils.mlflow_config import setup_mlflow
from utils.mlflow_run_manager import get_deployment_run
from utils.functions import predict_weather

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
    Endpoint for inference.
    Creates a new prediction run under the current model deployment run.
    """
    try:
        # Initialize response variables
        status = "success"
        message = "Daily prediction successfully completed"
        
        # Get deployment run for the current model
        # This function already calls setup_mlflow() internally
        deployment_run_id, model_version = get_deployment_run()
        
        # Create a nested run for this specific prediction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_run_name = f"prediction_{model_version}_{timestamp}"
        
        # Use the deployment run as parent
        with mlflow.start_run(run_id=deployment_run_id):
            with mlflow.start_run(run_name=prediction_run_name, nested=True) as nested_run:
                # Add tags to identify this run
                mlflow.set_tag("pipeline_type", "prediction")
                mlflow.set_tag("prediction_date", timestamp)
                mlflow.set_tag("endpoint", "predict_api")
                mlflow.set_tag("model_version", str(model_version))
                
                # Execute the prediction
                prediction, probability = predict_weather()
                
                return {
                    "status": status,
                    "message": message,
                    "run_id": nested_run.info.run_id,
                    "deployment_run_id": deployment_run_id,
                    "model_version": model_version,
                    "prediction": prediction,
                    "probability": probability
                }
                
    except Exception as e:
        # Clean up any active runs
        if mlflow.active_run():
            mlflow.end_run()
            
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))