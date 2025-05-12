"""
Tests unitaires pour les endpoints de l'API
"""
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest

# Vérifier que le monkey patching est bien effectué par conftest.py
import mlflow
print(f"MLflow in test_endpoint: {type(mlflow)}")
print(f"MLflow has set_registry_uri: {'set_registry_uri' in dir(mlflow)}")

# Importer l'application - devrait utiliser le mock MLflow
from main import app

# Création du client de test
client = TestClient(app)

def test_extract_endpoint():
    """Test de l'endpoint d'extraction"""
    # Patch supplémentaire pour cette fonction spécifique si nécessaire
    with patch('endpoint.extract_api.setup_mlflow', return_value=None):
        # Appel de l'endpoint avec un run_id (paramètre obligatoire)
        response = client.get("/extract?run_id=test_run_id")
        
        # Vérifications
        assert response.status_code == 200
        assert "status" in response.json()

def test_training_endpoint():
    """Test de l'endpoint d'entraînement"""
    # Patch supplémentaire pour cette fonction spécifique si nécessaire
    with patch('endpoint.training_api.setup_mlflow', return_value=None):
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
    
    # Patches supplémentaires pour cette fonction spécifique
    with patch('endpoint.predict_api.setup_mlflow', return_value=None), \
         patch('endpoint.predict_api.get_deployment_run', return_value=("test_deployment_run", "1")):
        
        # Appel de l'endpoint
        response = client.post("/predict_user", json=test_data)
        
        # Vérifications
        assert response.status_code == 200
        assert "prediction" in response.json()

def test_predict_automatic_endpoint():
    """Test de l'endpoint de prédiction automatique"""
    # Patches supplémentaires pour cette fonction spécifique
    with patch('endpoint.predict_api.setup_mlflow', return_value=None), \
         patch('endpoint.predict_api.get_deployment_run', return_value=("test_deployment_run", "1")), \
         patch('pathlib.Path.exists', return_value=True), \
         patch('pandas.read_csv', return_value=MagicMock(
             empty=False, 
             columns=["MinTemp", "MaxTemp", "Location"],
             iloc=MagicMock(
                 return_value=MagicMock(
                     to_dict=lambda: {"MinTemp": 10.0, "MaxTemp": 25.0, "Location": 1}
                 )
             )
         )):
        
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