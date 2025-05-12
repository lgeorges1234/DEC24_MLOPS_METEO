"""
Tests unitaires pour les endpoints de l'API
"""
# Chargement du mock MLflow AVANT toute importation
import sys
from unittest.mock import MagicMock, patch

# Assurer que notre mock MLflow est bien chargé avant toute autre importation
print("Initialisation des mocks MLflow pour les tests d'endpoint...")

# Ne pas essayer d'importer depuis tests_unitaires.mock_mlflow, définissons plutôt un mock local
mock_mlflow = MagicMock()
mock_mlflow.set_tracking_uri = MagicMock()
mock_mlflow.set_registry_uri = MagicMock()
mock_mlflow.get_tracking_uri = MagicMock(return_value="http://localhost:5000")
mock_mlflow.get_registry_uri = MagicMock(return_value="http://localhost:5000")
mock_mlflow.active_run = MagicMock(return_value=None)
mock_mlflow.tracking = MagicMock()
mock_mlflow.models = MagicMock()
mock_mlflow.sklearn = MagicMock()
mock_mlflow.pyfunc = MagicMock()

# Ajouter des comportements spécifiques
mock_client = MagicMock()
mock_client.get_model_version_by_alias = MagicMock(return_value=MagicMock(version="1", run_id="test_run_id"))
mock_mlflow.tracking.MlflowClient = MagicMock(return_value=mock_client)

# Fonction start_run simulée
def mock_start_run(*args, **kwargs):
    class MockRunContext:
        def __init__(self):
            self.info = MagicMock(run_id="test_run_id")
            self.data = MagicMock(tags={}, params={}, metrics={})
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
    
    return MockRunContext()

mock_mlflow.start_run = mock_start_run

# Mettre en place dans sys.modules
sys.modules['mlflow'] = mock_mlflow

# Patcher les fonctions critiques
with patch('utils.mlflow_run_manager.get_deployment_run', return_value=("test_deployment_run", "1")), \
     patch('utils.mlflow_config.setup_mlflow', return_value=None), \
     patch('utils.functions.predict_weather', return_value=(0, 0.85)):

    # Importer les modules après le mock
    from fastapi.testclient import TestClient
    from main import app

    # Création du client de test
    client = TestClient(app)

    def test_extract_endpoint():
        """Test de l'endpoint d'extraction"""
        # Appel de l'endpoint avec un run_id (paramètre obligatoire)
        response = client.get("/extract?run_id=test_run_id")
        
        # Vérifications
        assert response.status_code == 200
        assert "status" in response.json()

    def test_training_endpoint():
        """Test de l'endpoint d'entraînement"""
        # Appel de l'endpoint avec un run_id (paramètre obligatoire)
        response = client.get("/training?run_id=test_run_id")
        
        # Vérifications
        assert response.status_code == 200
        assert "status" in response.json()

    def test_predict_user_endpoint():
        """Test de l'endpoint de prédiction manuelle"""
        # Données de test
        test_data = {
            "Location": 1,
            "MinTemp": 10.0,
            "MaxTemp": 25.0,
            "WindGustDir": 180.0,
            "WindGustSpeed": 30.0,
            "WindDir9am": 180.0,
            "WindDir3pm": 180.0,
            "WindSpeed9am": 15.0,
            "WindSpeed3pm": 25.0,
            "Humidity9am": 70.0,
            "Humidity3pm": 50.0,
            "Pressure3pm": 1013.0,
            "Cloud9am": 3.0,
            "Cloud3pm": 5.0,
            "RainToday": 0
        }
        
        # Appel de l'endpoint
        response = client.post("/predict_user", json=test_data)
        
        # Vérifications
        assert response.status_code == 200
        assert "prediction" in response.json()

    def test_predict_automatic_endpoint():
        """Test de l'endpoint de prédiction automatique"""
        # Appel de l'endpoint
        response = client.get("/predict")
        
        # Vérifications
        assert response.status_code == 200
        assert "prediction" in response.json()

    def test_invalid_predict_user_input():
        """Test avec des données invalides"""
        invalid_data = {
            "Location": 1,
            "MinTemp": 100.0,  # Température trop élevée
            "MaxTemp": 25.0    # MaxTemp < MinTemp, devrait échouer
        }
        
        response = client.post("/predict_user", json=invalid_data)
        assert response.status_code == 422