'''
Endpoint pour l'extraction des données 
'''
# extract_api.py
import logging
from fastapi import APIRouter, HTTPException
from utils.functions import extract_and_prepare_df

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/extract")
async def extract_data() :
    """
    Endpoint pour l'extraction des données.
    """
    try:
        logger.info("Début de l'extraction des données")
        df, lencoders, clean_data_path = extract_and_prepare_df()

        return {
            "status": "Ok",
            "message": "Données extraites",
            "data_path": clean_data_path
        }
    except Exception as e:
        logger.error("Erreur lors de l'extraction: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
