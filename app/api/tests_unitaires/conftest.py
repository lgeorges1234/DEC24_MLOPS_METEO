"""
Configuration des tests avec pytest.
"""
import pytest
from unittest.mock import patch, MagicMock
import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Import notre module de mock MLflow avant tout autre import
# Cela assure que tout import de MLflow dans le code utilisera nos mocks
from tests_unitaires.mock_mlflow import mock_mlflow

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Configure l'environnement de test avec les variables d'environnement nécessaires"""
    # Configuration des variables d'environnement MLflow
    os.environ["MLFLOW_TRACKING_URI"] = "http://fake-mlflow:5000"
    os.environ["MLFLOW_REGISTRY_URI"] = "http://fake-mlflow:5000"
    os.environ["MLFLOW_MAX_RETRIES"] = "1"
    os.environ["MLFLOW_RETRY_DELAY"] = "0"
    os.environ["MLFLOW_REQUEST_TIMEOUT"] = "1"
    
    # Assurer que les répertoires existent
    os.makedirs("/app/api/data/models", exist_ok=True)
    os.makedirs("/app/api/data/prepared_data", exist_ok=True)
    os.makedirs("/app/api/data/metrics", exist_ok=True)
    os.makedirs("/app/raw_data/prediction_raw_data", exist_ok=True)
    
    yield
    
@pytest.fixture(autouse=True)
def check_model_files():
    """Vérifie si les fichiers modèles existent et les crée si nécessaire"""
    import os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    model_dir = "/app/api/data/models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Répertoire créé: {model_dir}")
    
    files = os.listdir(model_dir) if os.path.exists(model_dir) else []
    print(f"Fichiers dans {model_dir}: {files}")
    
    # Création ou vérification des fichiers nécessaires
    model_files = {
        "rfc.joblib": lambda: RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42),
        "scaler.joblib": lambda: StandardScaler(),
        "feature_order.joblib": lambda: [
            'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 
            'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure3pm',
            'Cloud9am', 'Cloud3pm', 'Location_encoded', 'WindGustDir_encoded',
            'WindDir9am_encoded', 'WindDir3pm_encoded', 'RainToday_encoded'
        ]
    }
    
    for filename, creator in model_files.items():
        filepath = os.path.join(model_dir, filename)
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            print(f"Création du fichier {filepath}")
            joblib.dump(creator(), filepath)
        else:
            print(f"Le fichier {filepath} existe et n'est pas vide")
            print(f"Taille: {os.path.getsize(filepath)} octets")
            
    # Créer un fichier de prédiction quotidienne pour les tests
    pred_file = "/app/raw_data/prediction_raw_data/daily_row_prediction.csv"
    if not os.path.exists(pred_file) or os.path.getsize(pred_file) == 0:
        print(f"Création du fichier de prédiction {pred_file}")
        test_data = {
            "Date": ["2023-01-01"],
            "Location": ["TestLocation"],
            "MinTemp": [15.0],
            "MaxTemp": [25.0],
            "Rainfall": [0.0],
            "Evaporation": [5.0],
            "Sunshine": [8.0],
            "WindGustDir": ["N"],
            "WindGustSpeed": [30.0],
            "WindDir9am": ["N"],
            "WindDir3pm": ["N"],
            "WindSpeed9am": [15.0],
            "WindSpeed3pm": [25.0],
            "Humidity9am": [70.0],
            "Humidity3pm": [50.0],
            "Pressure9am": [1015.0],
            "Pressure3pm": [1013.0],
            "Cloud9am": [3.0],
            "Cloud3pm": [5.0],
            "Temp9am": [18.0],
            "Temp3pm": [23.0],
            "RainToday": ["No"],
            "RainTomorrow": ["No"]
        }
        pd.DataFrame(test_data).to_csv(pred_file, index=False)
            
    yield

@pytest.fixture(autouse=True)
def mock_application_functions():
    """
    Mock des fonctions spécifiques à l'application.
    Cette fixture sera automatiquement appliquée à tous les tests.
    """    
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
        patch('utils.functions.predict_weather', return_value=(0, 0.85)), \
        patch('utils.mlflow_run_manager.get_deployment_run', return_value=("test_run_id", "1")):
        yield