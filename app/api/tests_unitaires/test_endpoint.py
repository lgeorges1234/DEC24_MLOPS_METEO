"""
Tests unitaires pour les endpoints de l'API - version compatible Docker
"""
import pytest
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Définir les variables d'environnement MLflow pour les tests
os.environ["MLFLOW_TRACKING_URI"] = "http://fake-mlflow:5000"
os.environ["MLFLOW_REGISTRY_URI"] = "http://fake-mlflow:5000"  # Ajout crucial

# Appliquer des patches pour les fonctions MLflow avant d'importer l'app
patches = []

# Patch pour setup_mlflow qui est appelé dans de nombreux endroits
@pytest.fixture(scope="module", autouse=True)
def mock_mlflow_setup():
    with patch('utils.mlflow_config.setup_mlflow', return_value=None) as mock:
        yield mock

# Patch pour get_deployment_run qui est appelé dans les endpoints de prédiction
@pytest.fixture(scope="module", autouse=True)
def mock_deployment_run():
    with patch('utils.mlflow_run_manager.get_deployment_run', return_value=("test-deployment-run", "1")) as mock:
        yield mock

# Patch pour predict_weather qui est appelé pour faire les prédictions
@pytest.fixture(scope="module", autouse=True)
def mock_predict_weather():
    with patch('utils.functions.predict_weather', return_value=(0, 0.85)) as mock:
        yield mock

# Patch pour Path.exists pour simuler l'existence des fichiers de prédiction
@pytest.fixture(scope="function")
def mock_file_exists():
    with patch('pathlib.Path.exists', return_value=True) as mock:
        yield mock

# Patch pour pandas.read_csv pour simuler la lecture des fichiers
@pytest.fixture(scope="function")
def mock_read_csv():
    with patch('pandas.read_csv', return_value=MagicMock(
        empty=False,
        iloc=MagicMock(return_value=MagicMock(
            to_dict=lambda: {"MinTemp": 10.0, "MaxTemp": 25.0, "Location": 1}
        )),
        columns=["MinTemp", "MaxTemp", "Location", "RainTomorrow"]
    )) as mock:
        yield mock

# Maintenant importer l'app après les patches
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

def test_predict_automatic_endpoint(mock_file_exists, mock_read_csv):
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