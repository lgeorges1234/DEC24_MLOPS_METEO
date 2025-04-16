
from fastapi.testclient import TestClient
from main import app  # Import de l'application

# Création du client de test
client = TestClient(app)

def test_extract_endpoint():
    """Test de l'endpoint d'extraction"""
    response = client.get("/extract")
    assert response.status_code == 200
    assert "status" in response.json()

def test_training_endpoint():
    """Test de l'endpoint d'entraînement"""
    response = client.post("/training")
    assert response.status_code == 200
    assert "model_path" in response.json()

def test_predict_user_endpoint():
    """Test de l'endpoint de prédiction manuelle"""
    # Exemple de données de test
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
    
    response = client.post("/predict_user", json=test_data)
    assert response.status_code == 200
    
    # Vérifiez la structure de la réponse
    json_response = response.json()
    assert "prediction" in json_response
    assert "probability" in json_response
    assert json_response["status"] == "success"

def test_predict_automatic_endpoint():
    """Test de l'endpoint de prédiction automatique"""
    response = client.get("/predict")
    assert response.status_code == 200
    
    json_response = response.json()
    assert "prediction" in json_response
    assert "probability" in json_response
    assert json_response["status"] == "success"

def test_invalid_predict_user_input():
    """Test avec des données invalides"""
    invalid_data = {
        "Location": 1,
        "MinTemp": 100.0, 
        "MaxTemp": 25.0
    }
    
    response = client.post("/predict_user", json=invalid_data)
    assert response.status_code == 422 