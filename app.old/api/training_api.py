# training_api.py
'''
API de l'entrainement des données
Elle est pour automatiser l'entrainement du modèle avec de nouvelles données
'''
import logging
from fastapi import APIRouter, HTTPException
from utils.functions import train_model

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/training")
async def train():
    """
    Endpoint pour l'entraînement du modèle.
    """
    try:
        saved_files = train_model()
        
        return {
            "status": "success",
            "message": "Modèle entraîné avec succès",
            "saved_files": saved_files
        }
    except Exception as e:
        logger.error("Erreur lors de l'entraînement: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
