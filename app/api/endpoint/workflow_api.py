"""
API for managing workflows that span multiple steps
"""
import logging
import requests
from endpoint.evaluate_api import evaluate
from endpoint.extract_api import extract
from endpoint.training_api import train
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
        workflow_response = await start_workflow()
        run_id = workflow_response["run_id"]
        
        try:
            # Step 1: Extract
            logger.info(f"Calling extract endpoint with run_id {run_id}")
            extract_response = await extract(run_id=run_id)
            
            # Step 2: Train
            logger.info(f"Calling training endpoint with run_id {run_id}")
            train_response = await train(run_id=run_id)
            
            # Step 3: Evaluate
            logger.info(f"Calling evaluate endpoint with run_id {run_id}")
            evaluate_response = await evaluate(run_id=run_id)
            
            # Mark workflow as completed
            await complete_workflow(run_id=run_id, status="COMPLETED")
            
            return {
                "status": "success",
                "message": "Complete workflow executed successfully",
                "run_id": run_id,
                "extract_result": extract_response,
                "train_result": train_response,
                "evaluate_result": evaluate_response
            }
        
        except Exception as e:
            # Mark workflow as failed and re-raise
            error_message = str(e)
            await complete_workflow(run_id=run_id, status="FAILED", error_message=error_message)
            raise
    
    except Exception as e:
        logger.error(f"Error during complete workflow execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))