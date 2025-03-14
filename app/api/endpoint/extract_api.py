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
async def extract(run_id: str = Query(None, description="ID of an existing MLflow workflow run")):
    """
    Endpoint pour l'extraction et la préparation des données.
    
    Si run_id est fourni, cette étape sera enregistrée dans une exécution MLflow existante.
    Sinon, une nouvelle exécution MLflow sera créée.
    """
    try:
        if run_id:
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
        else:
            # No run_id provided, use standalone run (as before)
            import mlflow
            from utils.mlflow_config import setup_mlflow
            from datetime import datetime
            
            # Ensure MLflow is properly set up and any existing runs are closed
            setup_mlflow()
            
            # Add timestamp to run name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"data_extraction_{timestamp}"
            
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("pipeline_type", "extraction_standalone")
                
                # Call extract_and_prepare_df WITH the required arguments
                df, encoders, output_file = extract_and_prepare_df(
                    path_raw_data=TRAINING_RAW_DATA_PATH,
                    csv_file=csv_file_training,
                    log_to_mlflow=True
                )
                
                mlflow.set_tag("extraction_status", "COMPLETED")
                
                return {
                    "status": "success",
                    "message": "Extraction et préparation terminées (run unique)",
                    "output_file": output_file,
                    "data_shape": str(df.shape)
                }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e