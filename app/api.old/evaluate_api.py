'''
endpoint Evaluation
'''
# evaluate_api.py
import logging
from fastapi import APIRouter, HTTPException
from utils.functions import evaluate_model

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/evaluate")
async def evaluate():
    """
    Endpoint d'évaluation le modèle.
    """
    try:
        metrics = evaluate_model()
        return {
            "status": "réussi",
            "message": "Évaluation terminée",
            "metrics": metrics
        }
    except Exception as e:
        logger.error("Erreur lors de l'évaluation: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
