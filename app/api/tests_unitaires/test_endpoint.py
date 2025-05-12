"""
Tests unitaires pour les endpoints de l'API
"""
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch, MagicMock  # Added this import
import pandas as pd  # Added this import

# Assurer que notre mock MLflow est chargé
import tests_unitaires.mock_mlflow

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
    
    # Mock all the necessary functions
    with patch('endpoint.predict_api.predict_weather', return_value=(0, 0.85)):
        # Appel de l'endpoint
        response = client.post("/predict_user", json=test_data)
        
        # Vérifications
        assert response.status_code == 200
        assert "prediction" in response.json()

def test_predict_automatic_endpoint():
    """Test de l'endpoint de prédiction automatique"""
    # Setup mock data
    mock_df = pd.DataFrame({
        "Location": ["Sydney"],
        "MinTemp": [15.0],
        "MaxTemp": [25.0],
        "Rainfall": [0.0],
        "WindGustDir": ["N"],
        "WindGustSpeed": [30.0],
        "WindDir9am": ["N"],
        "WindDir3pm": ["N"],
        "WindSpeed9am": [15.0],
        "WindSpeed3pm": [25.0],
        "Humidity9am": [70.0],
        "Humidity3pm": [50.0],
        "Pressure3pm": [1013.0],
        "Cloud9am": [3.0],
        "Cloud3pm": [5.0],
        "RainToday": ["No"]
    })
    
    # Directly mock at the endpoint level
    with patch('endpoint.predict_api.predict_weather', return_value=(0, 0.85)), \
         patch('pathlib.Path.exists', return_value=True), \
         patch('pandas.read_csv', return_value=mock_df):
         
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