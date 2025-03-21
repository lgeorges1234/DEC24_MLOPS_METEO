# evaluate_api.py
'''
API d'évaluation du modèle
'''
import logging
from utils.functions import evaluate_model
import mlflow
from fastapi import APIRouter, HTTPException, Query
# from utils.functions import evaluate_model_within_run
from utils.mlflow_config import setup_mlflow
from utils.mlflow_run_manager import complete_workflow_run, continue_workflow_run

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/evaluate")
async def evaluate(run_id: str = Query(..., description="ID of an existing MLflow workflow run")):
    """
    Endpoint pour l'évaluation du modèle.
    
    Un run_id DOIT être fourni pour enregistrer cette étape dans une exécution MLflow existante.
    """
    try:
        # Continue an existing workflow run
        with continue_workflow_run(run_id, "evaluation"):
            logger.info(f"Continuing workflow run {run_id} for evaluation step")
            
            # Add tags to identify this run
            mlflow.set_tag("pipeline_type", "evaluation")
            mlflow.set_tag("stage", "full_pipeline")
            mlflow.set_tag("endpoint", "evaluate_api")
            
            # Execute the evaluation
            metrics = evaluate_model()
            
            return {
                "status": "success",
                "message": "Évaluation terminée avec succès (workflow run)",
                "run_id": run_id,
                "metrics": metrics
            }
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation: {str(e)}")
        # If an exception occurs, mark the workflow as failed
        try:
            complete_workflow_run(run_id, "FAILED", str(e))
        except Exception as e2:
            logger.error(f"Erreur lors de la finalisation du workflow: {str(e2)}")
        raise HTTPException(status_code=500, detail=str(e))