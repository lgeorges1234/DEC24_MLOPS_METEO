"""
API pour la gestion de l'entraînement des modèles
"""
import logging
import mlflow
from fastapi import APIRouter, HTTPException, Request, Query
from typing import Optional
from utils.mlflow_run_manager import start_workflow_run, complete_workflow_run
from utils.mlflow_config import setup_mlflow
from utils.functions import train_model

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/training")
async def train(run_id: Optional[str] = Query(None, description="ID of an existing MLflow run")):
    """
    Endpoint for model training.
    
    If run_id is provided, this step will be recorded in an existing MLflow run.
    Otherwise, a new MLflow run will be created.
    """
    try:
        # Initialize response variables
        status = "success"
        message = "Model successfully trained"
        saved_files = None
        final_run_id = None
        
        # MLflow setup
        setup_mlflow()
        
        # run_id management
        if not run_id:
            # Create a new run_id if none is provided
            run_id = start_workflow_run("model_training")
            logger.info(f"Creating a new workflow for training: {run_id}")
        else:
            logger.info(f"Using existing workflow for training: {run_id}")
        
        # Check if there's already an active MLflow run
        active_run = mlflow.active_run()
        
        # Start a new MLflow run or use the existing one
        if active_run is None:
            # No active run, start a new one with the run_id
            with mlflow.start_run(run_id=run_id) as run:
                logger.info(f"Starting a new MLflow run with run_id: {run_id}")
                
                # Add tags to identify this run
                mlflow.set_tag("pipeline_type", "training")
                mlflow.set_tag("stage", "full_pipeline")
                mlflow.set_tag("endpoint", "training_api")
                
                # Execute the training
                saved_files = train_model()
                final_run_id = run_id
        else:
            # An execution is already active, check if it's the one we want
            if active_run.info.run_id == run_id:
                logger.info(f"Using existing active MLflow run: {run_id}")
            else:
                logger.warning(f"A different MLflow run is active ({active_run.info.run_id}). Using this one instead of {run_id}.")
                run_id = active_run.info.run_id
            
            # Add tags to the existing run
            mlflow.set_tag("pipeline_type", "training")
            mlflow.set_tag("stage", "full_pipeline")
            mlflow.set_tag("endpoint", "training_api")
            
            # Execute the training
            saved_files = train_model()
            final_run_id = run_id
            message += " (existing run)"
        
        # Single return statement
        return {
            "status": status,
            "message": message,
            "run_id": final_run_id,
            "saved_files": saved_files
        }
            
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        # If an exception occurs and we have a run_id, mark the workflow as failed
        if run_id:
            try:
                complete_workflow_run(run_id, "FAILED", str(e))
            except Exception as e2:
                logger.error(f"Error while finalizing the workflow: {str(e2)}")
        raise HTTPException(status_code=500, detail=str(e))
