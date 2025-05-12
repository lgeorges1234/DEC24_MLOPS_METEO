"""
Configuration des tests avec pytest.
"""
# IMPORTANT: Ces imports doivent être faits avant tout autre import pour patcher le module mlflow
import sys
from unittest.mock import MagicMock

# Créer un mock MLflow complet et le mettre en place AVANT les imports
class MockMLflow(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracking_uri = "http://localhost:5000"
        self._registry_uri = "http://localhost:5000"
        
    def set_tracking_uri(self, uri):
        self._tracking_uri = uri
        
    def set_registry_uri(self, uri):
        self._registry_uri = uri
        
    def get_tracking_uri(self):
        return self._tracking_uri
    
    def get_registry_uri(self):
        return self._registry_uri
    
    def active_run(self):
        return None
    
    def start_run(self, *args, **kwargs):
        # Créer un contexte simulé pour le with statement
        class MockRunContext:
            def __init__(self):
                self.info = MagicMock(run_id="test_run_id")
                self.data = MagicMock(tags={}, params={}, metrics={})
                
            def __enter__(self):
                return self
                
            def __exit__(self, *args):
                pass
        
        return MockRunContext()
    
    def end_run(self):
        pass

# Créer et installer le mock
mock_mlflow = MockMLflow()
mock_mlflow.tracking = MagicMock()
mock_mlflow.tracking.MlflowClient = MagicMock()
mock_mlflow.models = MagicMock()
mock_mlflow.models.signature = MagicMock()
mock_mlflow.models.signature.infer_signature = MagicMock(return_value=MagicMock())
mock_mlflow.sklearn = MagicMock()
mock_mlflow.sklearn.log_model = MagicMock(return_value=MagicMock(model_uri="mock://model/uri"))
mock_mlflow.sklearn.load_model = MagicMock(return_value=MagicMock(
    predict=lambda x: [0],
    predict_proba=lambda x: [[0.25, 0.75]]
))

# Créer des fonctions pour les méthodes spéciales du client
class MockClient(MagicMock):
    def get_model_version_by_alias(self, name, alias):
        return MagicMock(name=name, version="1", run_id="test_run_id")
    
    def set_registered_model_alias(self, name, alias, version):
        return None

# Configurer le client
mock_mlflow.tracking.MlflowClient.return_value = MockClient()

# Remplacer le module mlflow dans sys.modules
sys.modules['mlflow'] = mock_mlflow
sys.modules['mlflow.tracking'] = mock_mlflow.tracking
sys.modules['mlflow.models'] = mock_mlflow.models
sys.modules['mlflow.models.signature'] = mock_mlflow.models.signature
sys.modules['mlflow.sklearn'] = mock_mlflow.sklearn
sys.modules['mlflow.pyfunc'] = MagicMock()
print("MLflow mock installé avec succès au niveau système")

# Maintenant les imports standard peuvent se faire
import pytest
import os
import pandas as pd
import numpy as np

@pytest.fixture(autouse=True)
def check_model_files():
    """Vérifie si les fichiers modèles existent"""
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
    
    # Patcher directement les fonctions avec unittest.mock.patch
    from unittest.mock import patch
    
    # Patch extrait_and_prepare_df
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
        
        # Patcher get_deployment_run() est crucial pour predict_user et predict
        with patch('utils.mlflow_run_manager.get_deployment_run', return_value=("test_deployment_run", "1")):
            yield