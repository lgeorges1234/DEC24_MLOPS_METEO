"""
API for managing workflows that span multiple steps
"""
import logging
import requests
from fastapi import APIRouter, HTTPException, Request, Query
from utils.mlflow_run_manager import start_workflow_run, continue_workflow_run, complete_workflow_run

router = APIRouter()
logger = logging.getLogger(__name__)

@router.api_route("/workflow/start", methods=["GET", "POST"])
async def start_workflow():
    """
    Starts a new MLflow workflow and returns the run_id.
    This run_id can be used in subsequent API calls to connect the steps.
    """
    try:
        # Use the existing utility from mlflow_run_manager
        run_id = start_workflow_run("weather_prediction_workflow")
        
        return {
            "status": "success",
            "message": "Nouveau workflow démarré",
            "run_id": run_id,
            "next_steps": [
                {"endpoint": "/extract", "method": "GET", "params": {"run_id": run_id}},
                {"endpoint": "/training", "method": "GET", "params": {"run_id": run_id}},
                {"endpoint": "/evaluate", "method": "GET", "params": {"run_id": run_id}}
            ]
        }
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflow/complete")
async def complete_workflow(run_id: str, status: str = "COMPLETED", error_message: str = None):
    """
    Marks a workflow as completed or failed.
    """
    try:
        complete_workflow_run(run_id, status, error_message)
        return {
            "status": "success",
            "message": f"Workflow {run_id} marqué comme {status}",
            "run_id": run_id
        }
    except Exception as e:
        logger.error(f"Erreur lors de la finalisation du workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflow/run-full")
async def run_full_workflow():
    """
    Executes a complete workflow from start to finish.
    This includes extraction, training, and evaluation.
    """
    try:
        # Start a new workflow
        run_id = start_workflow_run("full_weather_pipeline")
        
        try:
            # Step 1: Extract
            logger.info(f"Calling extract endpoint with run_id {run_id}")
            extract_response = requests.get(
                "http://localhost:8000/extract", 
                params={"run_id": run_id}
            )
            extract_response.raise_for_status()
            
            # Step 2: Train
            logger.info(f"Calling training endpoint with run_id {run_id}")
            train_response = requests.get(
                "http://localhost:8000/training", 
                params={"run_id": run_id}
            )
            train_response.raise_for_status()
            
            # Step 3: Evaluate
            logger.info(f"Calling evaluate endpoint with run_id {run_id}")
            evaluate_response = requests.get(
                "http://localhost:8000/evaluate", 
                params={"run_id": run_id}
            )
            evaluate_response.raise_for_status()
            
            # Mark workflow as completed
            complete_workflow_run(run_id, "COMPLETED")
            
            return {
                "status": "success",
                "message": "Workflow complet exécuté avec succès",
                "run_id": run_id,
                "extract_result": extract_response.json(),
                "train_result": train_response.json(),
                "evaluate_result": evaluate_response.json()
            }
        
        except Exception as e:
            # Mark workflow as failed and re-raise
            error_message = str(e)
            complete_workflow_run(run_id, "FAILED", error_message)
            raise
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du workflow complet: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))