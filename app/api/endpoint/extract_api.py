# extract_api.py
'''
API d'extraction et de préparation des données
'''
import logging
from fastapi import APIRouter, HTTPException, Query
from utils.functions import extract_and_prepare_df, TRAINING_RAW_DATA_PATH, csv_file_training
from utils.mlflow_run_manager import continue_workflow_run

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/extract")
async def extract(run_id: str = Query(..., description="ID of an existing MLflow workflow run")):
    """
    Endpoint pour l'extraction et la préparation des données.
    
    Un run_id DOIT être fourni pour enregistrer cette étape dans une exécution MLflow existante.
    """
    try:
        # Continue an existing workflow run
        with continue_workflow_run(run_id, "extract"):
            logger.info(f"Continuing workflow run {run_id} for extraction step")
            
            # Call extract_and_prepare_df WITH the required arguments
            df, encoders, output_file = extract_and_prepare_df(
                path_raw_data=TRAINING_RAW_DATA_PATH,
                csv_file=csv_file_training,
                log_to_mlflow=True
            )
            
            return {
                "status": "success",
                "message": "Extraction et préparation terminées (workflow run)",
                "run_id": run_id,
                "output_file": output_file,
                "data_shape": str(df.shape)
            }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction: {str(e)}")
        # If an exception occurs, mark the workflow as failed
        try:
            from utils.mlflow_run_manager import complete_workflow_run
            complete_workflow_run(run_id, "FAILED", str(e))
        except Exception as e2:
            logger.error(f"Error while finalizing the workflow: {str(e2)}")
        raise HTTPException(status_code=500, detail=str(e)) from e