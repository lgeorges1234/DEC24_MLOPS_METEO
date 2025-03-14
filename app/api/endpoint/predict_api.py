# predict_api.py
'''
API de predection
Elle utilise le modele present dans le modele registry de MLflow
pour transmettre une prediction soit automatiquement en utilisant la ligne de prediction du jour - methode GET -
soit a traver un fichier csv fournit en POST request.
'''
from http.client import HTTPException
import logging
import mlflow
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
from pydantic import BaseModel
from utils.mlflow_run_manager import complete_workflow_run, continue_workflow_run, start_workflow_run
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
async def automatic_predict(run_id: Optional[str] = Query(None, description="ID of an existing MLflow run")):
    """
    Endpoint for inference.
    
    If run_id is provided, this step will be recorded in an existing MLflow run.
    Otherwise, a new MLflow run will be created.
    """
    try:
        # Initialize response variables
        status = "success"
        message = "Daily prediction successfully completed"
        prediction_result = None
        final_run_id = None
        
        # MLflow setup
        
        # run_id management
        if not run_id:
            # Create a new run_id if none is provided
            run_id = start_workflow_run("automatic_prediction")
            logger.info(f"Creating a new workflow for automatic prediction: {run_id}")
        else:
            logger.info(f"Using existing workflow for automatic prediction: {run_id}")
        
        # Check if there's already an active MLflow run
        active_run = mlflow.active_run()

        # Start a new MLflow run or use the existing one
        if active_run is None:
            # No active run, start a new one with the run_id
            with mlflow.start_run(run_id=run_id) as run:
                logger.info(f"Starting a new MLflow run with run_id: {run_id}")
                
                # Add tags to identify this run
                mlflow.set_tag("pipeline_type", "prediction")
                mlflow.set_tag("stage", "full_pipeline")
                mlflow.set_tag("endpoint", "predict_api")
                
                # Execute the prediction
                prediction, probability = predict_weather()
                prediction_result = (prediction, probability)
                final_run_id = run_id
                message += " (new workflow run)"
        else:
            # An execution is already active, check if it's the one we want
            if active_run.info.run_id == run_id:
                logger.info(f"Using existing active MLflow run: {run_id}")
            else:
                logger.warning(f"A different MLflow run is active ({active_run.info.run_id}). Using this one instead of {run_id}.")
                run_id = active_run.info.run_id
            
            # Add tags to the existing run
            mlflow.set_tag("pipeline_type", "prediction")
            mlflow.set_tag("stage", "full_pipeline")
            mlflow.set_tag("endpoint", "predict_api")
            
            # Call to the predict_weather function with MLflow context
            prediction, probability = predict_weather()
            prediction_result = (prediction, probability)
            final_run_id = run_id
            message += " (existing workflow run)"
        
        # Single return statement
        return {
            "status": status,
            "message": message,
            "run_id": final_run_id,
            "prediction": prediction_result[0],
            "probability": prediction_result[1]
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        # If an exception occurs and we have a run_id, mark the workflow as failed
        if run_id:
            try:
                complete_workflow_run(run_id, "FAILED", str(e))
            except Exception as e2:
                logger.error(f"Error while finalizing the workflow: {str(e2)}")
        raise HTTPException(status_code=500, detail=str(e))