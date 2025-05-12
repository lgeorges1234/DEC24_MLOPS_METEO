"""
Tests unitaires pour les endpoints de l'API
"""
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Définir la variable d'environnement TESTING avant d'importer l'application
os.environ["TESTING"] = "true"

# Importer l'application maintenant que TESTING est défini
from main import app

# Création du client de test
client = TestClient(app)

# Fixture pour simuler les fichiers de prédiction
@pytest.fixture
def mock_file_operations():
    with patch('pathlib.Path.exists', return_value=True), \
         patch('pandas.read_csv', return_value=MagicMock(
             empty=False,
             shape=(1, 10),
             columns=["MinTemp", "MaxTemp", "Location", "RainTomorrow"],
             iloc=MagicMock(
                 return_value=MagicMock(
                     to_dict=lambda: {"MinTemp": 10.0, "MaxTemp": 25.0, "Location": 1}
                 )
             )
         )):
        yield

# Fixture pour simuler extract_and_prepare_df
@pytest.fixture
def mock_extract_and_prepare():
    with patch('utils.functions.extract_and_prepare_df', return_value=(
            MagicMock(), # DataFrame mock
            {}, # encoders mock
            "/app/api/data/prepared_data/test_output.csv" # Output path
        )):
        yield

def test_extract_endpoint():
    """Test de l'endpoint d'extraction"""
    # Appel de l'endpoint avec un run_id (paramètre obligatoire)
    response = client.get("/extract?run_id=test_run_id")
    
    # Vérifications
    assert response.status_code == 200
    assert "status" in response.json()

def test_training_endpoint(mock_extract_and_prepare):
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
    
    # Patch spécifique pour predict_weather
    with patch('utils.functions.predict_weather', return_value=(0, 0.85)):
        # Appel de l'endpoint
        response = client.post("/predict_user", json=test_data)
        
        # Vérifications
        assert response.status_code == 200
        assert "prediction" in response.json()

def test_predict_automatic_endpoint(mock_file_operations):
    """Test de l'endpoint de prédiction automatique"""
    # Patch spécifique pour predict_weather
    with patch('utils.functions.predict_weather', return_value=(0, 0.85)):
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