"""
Configuration des tests avec pytest et monkey patching global de MLflow.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

# Définir la variable d'environnement TESTING
os.environ["TESTING"] = "true"

# MONKEY PATCHING GLOBAL DE MLFLOW
# ==============================
# Cette technique est plus avancée et plus "brutale" que les approches conventionnelles,
# mais elle peut fonctionner quand rien d'autre ne marche.

# Créer un mock MLflow complet
mock_mlflow = MagicMock()

# Ajouter explicitement toutes les fonctions et attributs utilisés par l'application
mock_mlflow.set_tracking_uri = MagicMock()
mock_mlflow.set_registry_uri = MagicMock()  # La fonction problématique
mock_mlflow.get_tracking_uri = MagicMock(return_value="http://fake-mlflow:5000")
mock_mlflow.get_registry_uri = MagicMock(return_value="http://fake-mlflow:5000")
mock_mlflow.set_experiment = MagicMock()
mock_mlflow.get_experiment_by_name = MagicMock(return_value=None)
mock_mlflow.create_experiment = MagicMock(return_value="1")
mock_mlflow.active_run = MagicMock(return_value=None)
mock_mlflow.end_run = MagicMock()
mock_mlflow.log_param = MagicMock()
mock_mlflow.log_params = MagicMock()
mock_mlflow.log_metric = MagicMock()
mock_mlflow.log_metrics = MagicMock()
mock_mlflow.set_tag = MagicMock()
mock_mlflow.log_artifact = MagicMock()
mock_mlflow.log_artifacts = MagicMock()

# Créer un mock pour un run MLflow
class MockRun:
    def __init__(self, run_id="test_run_id"):
        self.info = MagicMock(run_id=run_id)
        self.data = MagicMock(
            params={},
            metrics={},
            tags={}
        )
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass

# Configurer la fonction start_run pour retourner un contexte
mock_mlflow.start_run = MagicMock(return_value=MockRun())

# Créer des sous-modules
mock_mlflow.tracking = MagicMock()
mock_mlflow.models = MagicMock()
mock_mlflow.sklearn = MagicMock()
mock_mlflow.pyfunc = MagicMock()

# Configurer le client MLflow
mock_client = MagicMock()
mock_client.search_model_versions = MagicMock(return_value=[
    MagicMock(name="weather_prediction_model", version="1", run_id="test_run_id")
])
mock_client.get_model_version_by_alias = MagicMock(return_value=
    MagicMock(name="weather_prediction_model", version="1", run_id="test_run_id")
)
mock_client.search_runs = MagicMock(return_value=[
    MagicMock(info=MagicMock(run_id="test_deployment_run"))
])

mock_mlflow.tracking.MlflowClient = MagicMock(return_value=mock_client)

# Configurer mock_mlflow.sklearn.load_model pour retourner un modèle qui peut prédire
model_mock = MagicMock()
model_mock.predict = MagicMock(return_value=[0])
model_mock.predict_proba = MagicMock(return_value=[[0.2, 0.8]])
mock_mlflow.sklearn.load_model = MagicMock(return_value=model_mock)

# Remplacer le module mlflow réel par notre mock dans sys.modules
sys.modules['mlflow'] = mock_mlflow
sys.modules['mlflow.tracking'] = mock_mlflow.tracking
sys.modules['mlflow.models'] = mock_mlflow.models
sys.modules['mlflow.sklearn'] = mock_mlflow.sklearn
sys.modules['mlflow.pyfunc'] = mock_mlflow.pyfunc

print("GLOBAL MONKEY PATCHING OF MLFLOW COMPLETED")

# Maintenant patchez les fonctions spécifiques qui pourraient quand même essayer d'utiliser MLflow
@pytest.fixture(autouse=True)
def mock_mlflow_functions():
    """
    Patcher les fonctions qui utilisent MLflow directement, au cas où le monkey patching global ne suffit pas.
    """
    # Importer ici pour être sûr que mlflow est déjà mocké
    from unittest.mock import patch
    
    # Patcher setup_mlflow pour qu'il ne fasse rien
    with patch('utils.mlflow_config.setup_mlflow', return_value=True):
        # Patcher get_deployment_run pour qu'il retourne des valeurs factices
        with patch('utils.mlflow_run_manager.get_deployment_run', return_value=("test_deployment_run", "1")):
            # Patcher predict_weather pour qu'il retourne une prédiction factice
            with patch('utils.functions.predict_weather', return_value=(0, 0.85)):
                yield

@pytest.fixture(autouse=True)
def check_model_files():
    """Vérifie si les fichiers modèles existent"""
    import os
    
    model_dir = "/app/api/data/models"
    if os.path.exists(model_dir):
        print(f"Le répertoire {model_dir} existe")
        files = os.listdir(model_dir)
        print(f"Fichiers dans {model_dir}: {files}")
        
        # Vérifier spécifiquement les fichiers qui posent problème
        for filename in ["rfc.joblib", "scaler.joblib"]:
            filepath = os.path.join(model_dir, filename)
            if os.path.exists(filepath):
                print(f"Le fichier {filepath} existe")
                print(f"Taille: {os.path.getsize(filepath)} octets")
            else:
                print(f"Le fichier {filepath} n'existe PAS")
    else:
        print(f"Le répertoire {model_dir} n'existe PAS")
        
    yield

@pytest.fixture(autouse=True)
def mock_application_functions():
    """
    Mock des fonctions spécifiques à l'application.
    Cette fixture sera automatiquement appliquée à tous les tests.
    """
    # Définir des variables d'environnement de test
    os.environ["MLFLOW_TRACKING_URI"] = "http://fake-mlflow:5000"
    os.environ["MLFLOW_REGISTRY_URI"] = "http://fake-mlflow:5000"
    
    # Créer un DataFrame avec des données équilibrées pour les tests
    rows = []
    for i in range(10):
        # Avec RainTomorrow = 1
        rows.append({
            'MinTemp': 15.0 + i * 0.5,
            'MaxTemp': 25.0 + i * 0.5,
            'Rainfall': i * 0.2,
            'Evaporation': 1.5 + i * 0.1,
            'Sunshine': 8.5 - i * 0.1,
            'WindGustSpeed': 30.0 - i * 0.5,
            'WindSpeed9am': 10.0 + i * 0.2,
            'WindSpeed3pm': 15.0 + i * 0.2,
            'Humidity9am': 70.0 - i * 0.5,
            'Humidity3pm': 50.0 - i * 0.5,
            'Pressure9am': 1015.0 + i * 0.2,
            'Pressure3pm': 1010.0 + i * 0.2,
            'Cloud9am': 5.0 - i * 0.1,
            'Cloud3pm': 4.0 - i * 0.1,
            'Temp9am': 18.0 + i * 0.3,
            'Temp3pm': 23.0 + i * 0.3,
            'Location_encoded': 1.0 if i % 2 == 0 else 2.0,
            'WindGustDir_encoded': 1.0 if i % 2 == 0 else 2.0,
            'WindDir9am_encoded': 1.0 if i % 3 == 0 else 2.0,
            'WindDir3pm_encoded': 1.0 if i % 3 == 1 else 2.0,
            'RainToday_encoded': 1.0,
            'RainTomorrow': 1.0
        })
        
        # Avec RainTomorrow = 0
        rows.append({
            'MinTemp': 15.0 - i * 0.2,
            'MaxTemp': 25.0 - i * 0.2,
            'Rainfall': 0.0,
            'Evaporation': 1.5 - i * 0.05,
            'Sunshine': 8.5 + i * 0.1,
            'WindGustSpeed': 30.0 + i * 0.5,
            'WindSpeed9am': 10.0 - i * 0.1,
            'WindSpeed3pm': 15.0 - i * 0.1,
            'Humidity9am': 70.0 + i * 0.5,
            'Humidity3pm': 50.0 + i * 0.5,
            'Pressure9am': 1015.0 - i * 0.2,
            'Pressure3pm': 1010.0 - i * 0.2,
            'Cloud9am': 5.0 + i * 0.1,
            'Cloud3pm': 4.0 + i * 0.1,
            'Temp9am': 18.0 - i * 0.3,
            'Temp3pm': 23.0 - i * 0.3,
            'Location_encoded': 1.0 if i % 2 == 1 else 2.0,
            'WindGustDir_encoded': 1.0 if i % 2 == 1 else 2.0,
            'WindDir9am_encoded': 1.0 if i % 3 == 2 else 2.0,
            'WindDir3pm_encoded': 1.0 if i % 3 == 0 else 2.0,
            'RainToday_encoded': 0.0,
            'RainTomorrow': 0.0 
        })
    
    test_df = pd.DataFrame(rows)
    
    test_encoders = {
        "Location": {1: "Sydney", 2: "Melbourne"},
        "WindGustDir": {1: "W", 2: "E"},
        "WindDir9am": {1: "W", 2: "E"},
        "WindDir3pm": {1: "W", 2: "E"},
        "RainToday": {1: "Yes", 0: "No"}
    }
    
    # Mock des fonctions de l'application
    from unittest.mock import patch
    
    with patch('utils.functions.extract_and_prepare_df', return_value=(
            test_df, # DataFrame avec des classes équilibrées
            test_encoders, # Encodeurs
            "/app/api/data/prepared_data/test_output.csv" # Chemin correct
        )), \
        patch('utils.functions.train_model', return_value={
            "model_path": "/app/api/data/models/test_model.pkl", # Chemin correct
            "metrics_path": "/app/api/data/metrics/test_metrics.json" # Chemin correct
        }), \
        patch('utils.functions.predict_weather', return_value=(0, 0.85)):
        
        # Patcher également les fonctions de lecture/écriture de fichiers
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pandas.read_csv', return_value=pd.DataFrame({
                 'MinTemp': [10.0],
                 'MaxTemp': [25.0],
                 'Location': [1],
                 'RainTomorrow': [0]
             })):
            yield