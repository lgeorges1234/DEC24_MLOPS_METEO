"""
Configuration des tests avec pytest.
"""
import pytest
from unittest.mock import patch, MagicMock
import os
import sys
import pandas as pd
import numpy as np

# Import notre module de mock MLflow avant tout autre import
# Cela assure que tout import de MLflow dans le code utilisera nos mocks
from tests_unitaires.mock_mlflow import mock_mlflow

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
        yield