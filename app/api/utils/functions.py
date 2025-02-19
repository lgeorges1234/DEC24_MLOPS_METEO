'''
Liste des fonctions que nous appellerons dans les différents scripts
'''

# api/utils/functions.py
from pathlib import Path
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
import json

logger = logging.getLogger(__name__)

# Définition des chemins
BASE_PATH = Path(__file__).parent.parent.parent
INITIAL_DATA_PATH = BASE_PATH / "initial_dataset" / "weatherAUS.csv"
TRAINING_RAW_DATA_PATH = BASE_PATH / "training_raw_data"
PREDICTION_RAW_DATA_PATH = BASE_PATH / "prediction_raw_data"
CLEAN_DATA_PATH = BASE_PATH / "prepared_data"
METRICS_DATA_PATH = BASE_PATH / "metrics"
MODEL_PATH = BASE_PATH / "models"

# Create directories if they don't exist
for path in [TRAINING_RAW_DATA_PATH, PREDICTION_RAW_DATA_PATH, CLEAN_DATA_PATH, METRICS_DATA_PATH, MODEL_PATH]:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")


def outlier_thresholds(dataframe, column, q1 = 0.25, q3 = 0.75):
    """
    Equation outliers.
    """
    quartile1 = dataframe[column].quantile(q1)
    quartile3 = dataframe[column].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return up_limit, low_limit 

def replace_with_thresholds(dataframe, column):
    """
    Remplace les outliers par les valeurs limites.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, column)
    dataframe.loc[(dataframe[column] < low_limit), column] = low_limit
    dataframe.loc[(dataframe[column] > up_limit), column] = up_limit

def extract_and_prepare_df():
    """
    Extraction et préparation des données météorologiques à partir du .csv
    """
    try:
        # Extraction
        logger.info("Lecture du fichier: %s", TRAINING_RAW_DATA_PATH / "weatherAUS.csv")
        if not (TRAINING_RAW_DATA_PATH / "weatherAUS.csv").exists():
            raise FileNotFoundError(f"Le fichier {TRAINING_RAW_DATA_PATH / 'weatherAUS.csv'} n'a pas été trouvé")

        df = pd.read_csv(TRAINING_RAW_DATA_PATH / "weatherAUS.csv")
        logger.info("Données chargées")

        # Rest of the data preparation code remains the same
        # Conversion des variables catégorielles cibles en variables binaires
        df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
        df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})

        # Identification et séparation des colonnes catégorielles et continues
        categorical, continuous = [],[]

        for col in df.columns:
            if df[col].dtype == 'object':
                categorical.append(col)
            else:
                continuous.append(col)

        # Gestion des valeurs vides
        for col in categorical:
            df[col] = df[col].fillna(df[col].mode()[0])

        # Encodage des variables catégorielles
        lencoders = {}

        for col in df.select_dtypes(include=['object']).columns:
            lencoders[col] = LabelEncoder()
            df[col] = lencoders[col].fit_transform(df[col])

        # Suppression des lignes avec valeurs manquantes dans les variables cibles
        df = df.dropna(subset=['RainToday', 'RainTomorrow'])

        # Modification des seuils des variables cibles
        columns_for_outliers = df.drop(columns=['RainTomorrow', 'RainToday', 'Date', 'Location']).columns
        for column in columns_for_outliers:
            replace_with_thresholds(df, column)

        # Suppression des colonnes non nécessaires
        columns_to_drop = ['Date', 'Temp3pm', 'Pressure9am', 'Temp9am', 'Rainfall']
        df.drop(columns_to_drop, axis=1, inplace=True)

        logger.info("Préparation terminée. Dimensions finales: %s", df.shape)

        # Sauvegarde du fichier nettoyé
        df.to_csv(CLEAN_DATA_PATH / "meteo.csv", index=False)
        logger.info("Données nettoyées sauvegardées")

        return df, lencoders, str(CLEAN_DATA_PATH / "meteo.csv")

    except Exception as e:
        logger.error("Erreur lors du traitement des données: %s", str(e))
        raise

def train_model():
    """
    Entraînement du RandomForestClassifier
    """
    try:
        # Chargement des données
        logger.info("Chargement des données nettoyées")
        data = pd.read_csv(CLEAN_DATA_PATH / "meteo.csv")

        # Séparation de la variable cible des features
        X = data.drop(columns=["RainTomorrow"]).astype("float")
        y = data["RainTomorrow"]

        # Sauvegarde du nom des colonnes pour transformation du 'array' en DataFrame
        features_names = X.columns

        # Séparation du jeux d'entrainement et de tests
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Standardisation des données
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Sauvegarde des données d'entrainement et de tests
        saved_files = {}
        pd.DataFrame(X_train_scaled, columns=features_names).to_csv(CLEAN_DATA_PATH / "X_train.csv", index=False)
        pd.Series(y_train).to_csv(CLEAN_DATA_PATH / "y_train.csv", index=False)
        # Enregistrement des chemins pour garder une trace
        saved_files["train_data"] = {
            "X_train": str(CLEAN_DATA_PATH / "X_train.csv"),
            "y_train": str(CLEAN_DATA_PATH / "y_train.csv")
        }

        pd.DataFrame(X_test, columns=features_names).to_csv(CLEAN_DATA_PATH / "X_test.csv", index=False)
        pd.Series(y_test).to_csv(CLEAN_DATA_PATH / "y_test.csv", index=False)
        # Enregistrement des chemins pour garder une trace
        saved_files["test_data"] = {
            "X_test": str(CLEAN_DATA_PATH / "X_test.csv"),
            "y_test": str(CLEAN_DATA_PATH / "y_test.csv")
        }


        # Entraînement
        logger.info("Entraînement du modèle")
        params = {"n_estimators": 10, "max_depth": 10, "random_state": 42}
        rfc = RandomForestClassifier(**params)
        rfc.fit(X_train, y_train)

        # Sauvegarde du modèle et du scaler
        model_path = MODEL_PATH / "rfc.joblib"
        scaler_path = MODEL_PATH / "scaler.joblib"

        joblib.dump(rfc, model_path)
        joblib.dump(scaler, scaler_path)

        # Enregistrement des chemins pour garder une trace
        saved_files["model"] = str(model_path)
        saved_files["scaler"] = str(scaler_path)

        logger.info("Modèle entraîné et sauvegardés avec succès")
        return saved_files

    except Exception as e:
        logger.error("Erreur lors de l'entraînement: %s", str(e))
        raise

def evaluate_model():
    """
    Évaluation du modèle.
    """
    try:
        # Chargement des données de test
        logger.info("Chargement des données de test")
        X_test = pd.read_csv(CLEAN_DATA_PATH / "X_test.csv")
        X_train = pd.read_csv(CLEAN_DATA_PATH / "X_train.csv")
        y_test = pd.read_csv(CLEAN_DATA_PATH / "y_test.csv")["RainTomorrow"]
        y_train = pd.read_csv(CLEAN_DATA_PATH / "y_train.csv")["RainTomorrow"]

        # Chargement du modèle et du scaler
        model = joblib.load(MODEL_PATH / "rfc.joblib")
        scaler = joblib.load(MODEL_PATH / "scaler.joblib")

        # Préparation des données
        X_test_scaled = scaler.transform(X_test)

        # Prédictions sur les données de test
        y_test_pred = model.predict(X_test)
        y_test_probs = model.predict_proba(X_test_scaled)[:,1]

        # Prédiction sur les données d'entrainement (rappel: X_train est déjà normalisé)
        y_train_pred = model.predict(X_train)
        y_train_probs = model.predict_proba(X_train)[:,1]

        # Calcul des métriques
        metrics_rfc = {
            "test": {
                "accuracy": float(metrics.accuracy_score(y_test, y_test_pred)),
                "precision": float(metrics.precision_score(y_test, y_test_pred)),
                "recall": float(metrics.recall_score(y_test, y_test_pred)),
                "f1": float(metrics.f1_score(y_test, y_test_pred)),
                "roc_auc": float(metrics.roc_auc_score(y_test, y_test_probs))
                },
            "train": {
                "accuracy": float(metrics.accuracy_score(y_train, y_train_pred)),
                "precision": float(metrics.precision_score(y_train, y_train_pred)),
                "recall": float(metrics.recall_score(y_train, y_train_pred)),
                "f1": float(metrics.f1_score(y_train, y_train_pred)),
                "roc_auc": float(metrics.roc_auc_score(y_train, y_train_probs))
            }
        }

        logger.info("Évaluation terminée")

        # Sauvegarde des métriques
        with open(METRICS_DATA_PATH / "metrics.json", "w") as f:
            json.dump(metrics_rfc, f, indent=4)

        return metrics_rfc

    except Exception as e:
        logger.error("Erreur lors de l'évaluation: %s", str(e))
        raise

def predict_weather(data):
    """
    Prédiction sur de nouvelles données.
    """
    try:
        # Chargement du modèle et du scaler
        model = joblib.load(MODEL_PATH / "rfc.joblib")
        scaler = joblib.load(MODEL_PATH / "scaler.joblib")

        # Conversion en DataFrame
        input_df = pd.DataFrame([data])
        
        # Préparation des données
        input_scaled = scaler.transform(input_df)
        
        # Prédiction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        logger.info("Prédiction effectuée avec succès")
        return prediction, float(probability)

    except Exception as e:
        logger.error("Erreur lors de la prédiction: %s", str(e))
        raise