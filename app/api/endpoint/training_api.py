"""
API pour la gestion de l'entraînement des modèles
"""
import logging
import mlflow
from fastapi import APIRouter, HTTPException, Query
from utils.mlflow_run_manager import continue_workflow_run, complete_workflow_run
from utils.functions import train_model

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/training")
async def train(run_id: str = Query(..., description="ID of an existing MLflow workflow run")):
    """
    Endpoint for model training.
    
    A run_id MUST be provided to record this step in an existing MLflow workflow run.
    This prevents standalone runs from being created automatically.
    """
    try:
        # Continue an existing workflow run
        with continue_workflow_run(run_id, "training"):
            logger.info(f"Continuing workflow run {run_id} for training step")
            
            # Add tags to identify this run
            mlflow.set_tag("pipeline_type", "training")
            mlflow.set_tag("stage", "full_pipeline")
            mlflow.set_tag("endpoint", "training_api")
            
            # Execute the training
            saved_files = train_model()
            
            return {
                "status": "success",
                "message": "Model successfully trained (workflow run)",
                "run_id": run_id,
                "saved_files": saved_files
            }
            
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        # If an exception occurs, mark the workflow as failed
        try:
            complete_workflow_run(run_id, "FAILED", str(e))
        except Exception as e2:
            logger.error(f"Error while finalizing the workflow: {str(e2)}")
        raise HTTPException(status_code=500, detail=str(e))