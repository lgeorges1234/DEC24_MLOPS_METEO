# training_api.py
'''
API de l'entrainement des données
Elle est pour automatiser l'entrainement du modèle avec de nouvelles données
'''
import logging
from pathlib import Path
from fastapi import HTTPException, APIRouter
import joblib

# Configuration
router = APIRouter()
logging.basicConfig(level=logging.INFO)
MODEL_PATH = Path('../models/rfc.pkl')
logger = logging.getLogger(__name__)

'''
# Chemin des scripts au PATH de Python
sys.path.append("./src/data")  # A titre informatif:
-  Le __init__ est présent dans chaque dossier pour éviter d'utiliser sys.path
sys.path.append("./src/models") # A titre informatif:
- Le __init__ est présent dans chaque dossier pour éviter d'utiliser sys.path
'''
# Importation des scripts de préparation et d'entrainement
import prepare_data  # script de préparation
import train_model # script d'entrainement du modèle

def train_existing_model(existing_model):
    """
    Charge le modèle existant, le réentraîne et renvoie le modèle réentraîné
    ainsi que les métriques.
    """
    try:
        # Preparation des nouvelles données
        logger.info("Preparation des nouvelles données")
        prepare_data.prepare_data()

        # Réentraîner le modèle en appelant ton script d'entraînement existant
        logger.info("Réentraîner le modèle avec de nouvelles données")
        existing_model, metrics = train_model.train_model(existing_model)

        logger.info("Métriques après réentraînement: %s", metrics)
        return existing_model, metrics

    except Exception as e:
        logger.error("Erreur lors de l'entraînement: %s", str(e))
        raise

@router.post("/training")
async def training():
    """
    Endpoint qui exécute la préparation des données et l'entraînement du modèle
    à partir des scripts initiaux
    """
    try:

        '''
        # Chargement du dernier modèle selectionné (chemin à modifier avec le chemin du dernier modèle selectionné avec MLFlow)
        url = "chemin MLFlow"
        model = mlflow.pyfunc.load_model(url)
        print('Modèle chargé)
        '''
        #Chargement du modèle
        model_charged = joblib.load(MODEL_PATH)
        logger.info('Modèle chargé')

        # Exécution de l'entraînement
        logger.info("Début de l'entraînement")
        _, metrics = train_existing_model(model_charged)
        logger.info("Entraînement terminé")

        return {
            "status": "success",
            "message": "Préparation et entraînement terminés avec succès",
            "metrics": metrics
        }

    except Exception as e:
        logger.error("Erreur dans le processus: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du processus: {str(e)}"
        ) from e
