'''
Liste des fonctions que nous appellerons dans les différents scripts
Avec intégration MLflow pour le tracking et le registre de modèles
Approche de suivi avec une exécution unique par pipeline
'''

from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
import json
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
from utils.mlflow_config import setup_mlflow, MODEL_NAME

logger = logging.getLogger(__name__)

# Définition des chemins
BASE_PATH = Path(__file__).parent.parent.parent
TRAINING_RAW_DATA_PATH = Path("/app/raw_data/training_raw_data")
PREDICTION_RAW_DATA_PATH = Path("/app/raw_data/prediction_raw_data")
CLEAN_DATA_PATH = Path("/app/api/data/prepared_data")
METRICS_DATA_PATH = Path("/app/api/data/metrics")
MODEL_PATH = Path("/app/api/data/models")
csv_file_training = "weatherAUS_training.csv"
csv_file_prediction = "weatherAUS_prediction.csv"
csv_file_daily_prediction = "daily_row_prediction.csv"

# Create directories if they don't exist
for path in [CLEAN_DATA_PATH, METRICS_DATA_PATH, MODEL_PATH]:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Created directory: %s", path)


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

def extract_and_prepare_df(path_raw_data, csv_file, log_to_mlflow=True):
    """
    Extraction et préparation des données météorologiques à partir du .csv
    
    Args:
        path_raw_data: Chemin vers les données brutes
        csv_file: Nom du fichier CSV
        log_to_mlflow: Si True, enregistre les métriques et paramètres dans MLflow
    """
    try:
        # Extraction
        logger.info("Lecture du fichier: %s", path_raw_data / csv_file)
        if not (path_raw_data / csv_file).exists():
            raise FileNotFoundError(f"Le fichier {path_raw_data / csv_file} n'a pas été trouvé")

        df = pd.read_csv(path_raw_data / csv_file)
        logger.info("Données chargées")
        
        if log_to_mlflow:
            # Track initial data shape
            mlflow.log_param("data_preparation.input_file", str(path_raw_data / csv_file))
            mlflow.log_param("data_preparation.initial_shape", str(df.shape))

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

        if log_to_mlflow:
            # Log categorical and continuous columns
            mlflow.log_param("data_preparation.categorical_columns", categorical)
            mlflow.log_param("data_preparation.continuous_columns", continuous)

        # Gestion des valeurs vides
        missing_before = df.isnull().sum().sum()
        for col in categorical:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        if log_to_mlflow:
            # Log missing value handling
            missing_after = df.isnull().sum().sum()
            mlflow.log_metric("data_preparation.missing_values_before", missing_before)
            mlflow.log_metric("data_preparation.missing_values_after", missing_after)

        # Encodage des variables catégorielles
        lencoders = {}

        for col in df.select_dtypes(include=['object']).columns:
            lencoders[col] = LabelEncoder()
            df[col] = lencoders[col].fit_transform(df[col])

        # Suppression des lignes avec valeurs manquantes dans les variables cibles
        rows_before = len(df)
        df = df.dropna(subset=['RainToday', 'RainTomorrow'])
        rows_after = len(df)
        
        if log_to_mlflow:
            # Log row reduction
            mlflow.log_metric("data_preparation.rows_before_dropna", rows_before)
            mlflow.log_metric("data_preparation.rows_after_dropna", rows_after)
            mlflow.log_metric("data_preparation.rows_dropped", rows_before - rows_after)

        # Modification des seuils des variables cibles
        columns_for_outliers = df.drop(columns=['RainTomorrow', 'RainToday', 'Date', 'Location']).columns
        for column in columns_for_outliers:
            replace_with_thresholds(df, column)

        # Suppression des colonnes non nécessaires
        columns_to_drop = ['Date', 'Temp3pm', 'Pressure9am', 'Temp9am', 'Rainfall']
        df.drop(columns_to_drop, axis=1, inplace=True)
        
        if log_to_mlflow:
            # Log dropped columns
            mlflow.log_param("data_preparation.dropped_columns", columns_to_drop)

        logger.info("Préparation terminée. Dimensions finales: %s", df.shape)
        
        if log_to_mlflow:
            mlflow.log_param("data_preparation.final_shape", str(df.shape))

        # Définition du nom du fichier nettoyé
        cleaned_file_name = f"{Path(csv_file).stem}_cleaned.csv"

        # Sauvegarde du fichier nettoyé
        df.to_csv(CLEAN_DATA_PATH / cleaned_file_name, index=False)
        logger.info("Données nettoyées sauvegardées")
        
        if log_to_mlflow:
            # Log output file location
            mlflow.log_param("data_preparation.output_file", str(CLEAN_DATA_PATH / cleaned_file_name))

        return df, lencoders, str(CLEAN_DATA_PATH / cleaned_file_name)

    except Exception as e:
        logger.error("Erreur lors du traitement des données: %s", str(e))
        if log_to_mlflow and mlflow.active_run():
            mlflow.set_tag("data_preparation.status", "FAILED")
            mlflow.set_tag("data_preparation.error", str(e))
        raise

def train_model():
    """
    Entraînement du RandomForestClassifier avec suivi MLflow.
    Cette version simplifiée suppose qu'une exécution MLflow est déjà active.
    """
    try:
        saved_files = {}
        
        # 1. Préparation des données (avec logging MLflow)
        mlflow.set_tag("current_step", "data_preparation")
        logger.info("Extraction et préparation des données")

            # Input file for prediction
        input_file = TRAINING_RAW_DATA_PATH / csv_file_training
        mlflow.log_param("training.input_file", str(input_file))

        logger.info(f"training input file: {input_file}")

        df, lencoders, cleaned_file = extract_and_prepare_df(
            TRAINING_RAW_DATA_PATH, 
            csv_file_training,
            log_to_mlflow=True
        )
        
        mlflow.log_param("training.input_shape", str(df.shape))


        # 2. Chargement des données nettoyées
        mlflow.set_tag("current_step", "data_loading")
        logger.info("Chargement des données nettoyées")
        
        # Séparation de la variable cible des features
        X = df.drop(columns=["RainTomorrow"]).astype("float")
        y = df["RainTomorrow"]

        # Sauvegarde du nom des colonnes
        features_names = X.columns
        mlflow.log_param("training.features", list(features_names))

        # Paramètres pour la division des données
        test_size = 0.2
        random_state = 42
        mlflow.log_param("training.test_size", test_size)
        mlflow.log_param("training.random_state", random_state)
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        mlflow.log_metric("training.train_samples", len(X_train))
        mlflow.log_metric("training.test_samples", len(X_test))
        
        # Distribution des classes 
        train_class_dist = y_train.value_counts(normalize=True).to_dict()
        test_class_dist = y_test.value_counts(normalize=True).to_dict()
        
        for label, freq in train_class_dist.items():
            mlflow.log_metric(f"training.train_class_{int(label)}_freq", freq)
        
        for label, freq in test_class_dist.items():
            mlflow.log_metric(f"training.test_class_{int(label)}_freq", freq)

        # Standardisation des données
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Statistiques de scaling
        for i, feature in enumerate(features_names):
            mlflow.log_metric(f"training.scaler_mean_{feature}", scaler.mean_[i])
            mlflow.log_metric(f"training.scaler_scale_{feature}", scaler.scale_[i])

        # Sauvegarde des données
        pd.DataFrame(X_train_scaled, columns=features_names).to_csv(CLEAN_DATA_PATH / "X_train.csv", index=False)
        pd.Series(y_train).to_csv(CLEAN_DATA_PATH / "y_train.csv", index=False)
        
        saved_files["train_data"] = {
            "X_train": str(CLEAN_DATA_PATH / "X_train.csv"),
            "y_train": str(CLEAN_DATA_PATH / "y_train.csv")
        }

        pd.DataFrame(X_test, columns=features_names).to_csv(CLEAN_DATA_PATH / "X_test.csv", index=False)
        pd.Series(y_test).to_csv(CLEAN_DATA_PATH / "y_test.csv", index=False)
        
        saved_files["test_data"] = {
            "X_test": str(CLEAN_DATA_PATH / "X_test.csv"),
            "y_test": str(CLEAN_DATA_PATH / "y_test.csv")
        }
        
        # 3. Entraînement du modèle
        mlflow.set_tag("current_step", "model_training")
        logger.info("Entraînement du modèle")
        
        # Hyperparamètres
        params = {"n_estimators": 10, "max_depth": 10, "random_state": 42}
        mlflow.log_params(params)
        
        # Création et entraînement du modèle
        rfc = RandomForestClassifier(**params)
        rfc.fit(X_train_scaled, y_train)
        
        # Prédictions et évaluation
        train_preds = rfc.predict(X_train_scaled)
        X_test_scaled = scaler.transform(X_test)
        test_preds = rfc.predict(X_test_scaled)
        
        # Métriques d'entraînement
        train_accuracy = metrics.accuracy_score(y_train, train_preds)
        train_precision = metrics.precision_score(y_train, train_preds)
        train_recall = metrics.recall_score(y_train, train_preds)
        train_f1 = metrics.f1_score(y_train, train_preds)
        
        mlflow.log_metric("training.train_accuracy", train_accuracy)
        mlflow.log_metric("training.train_precision", train_precision)
        mlflow.log_metric("training.train_recall", train_recall)
        mlflow.log_metric("training.train_f1", train_f1)
        
        # Métriques de test
        test_accuracy = metrics.accuracy_score(y_test, test_preds)
        test_precision = metrics.precision_score(y_test, test_preds)
        test_recall = metrics.recall_score(y_test, test_preds)
        test_f1 = metrics.f1_score(y_test, test_preds)
        
        mlflow.log_metric("training.test_accuracy", test_accuracy)
        mlflow.log_metric("training.test_precision", test_precision)
        mlflow.log_metric("training.test_recall", test_recall)
        mlflow.log_metric("training.test_f1", test_f1)
        
        # Importance des caractéristiques
        for i, feature in enumerate(features_names):
            importance = float(rfc.feature_importances_[i])
            mlflow.log_metric(f"training.feature_importance_{feature}", importance)
        
        # Signature du modèle pour MLflow
        signature = infer_signature(X_train_scaled, rfc.predict(X_train_scaled))
        
        # Enregistrement du modèle dans MLflow
        model_info = mlflow.sklearn.log_model(
            sk_model=rfc,
            artifact_path="model",
            signature=signature,
            input_example=X_train_scaled[:5],
            registered_model_name=MODEL_NAME
        )
        
        # Sauvegarde locale (pour compatibilité)
        model_path = MODEL_PATH / "rfc.joblib"
        scaler_path = MODEL_PATH / "scaler.joblib"
        
        joblib.dump(rfc, model_path)
        joblib.dump(scaler, scaler_path)
        
        mlflow.log_param("training.local_model_path", str(model_path))
        mlflow.log_param("training.local_scaler_path", str(scaler_path))
        
        saved_files["model"] = str(model_path)
        saved_files["scaler"] = str(scaler_path)
        
        # Sauvegarder les métriques dans un fichier JSON (pour compatibilité)
        metrics_dict = {
            "train": {
                "accuracy": float(train_accuracy),
                "precision": float(train_precision),
                "recall": float(train_recall),
                "f1": float(train_f1)
            },
            "test": {
                "accuracy": float(test_accuracy),
                "precision": float(test_precision),
                "recall": float(test_recall),
                "f1": float(test_f1)
            }
        }
        
        metrics_file = METRICS_DATA_PATH / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        
        # Log le fichier de métriques comme artefact
        mlflow.log_artifact(str(metrics_file))
        
        # Statut final
        mlflow.set_tag("pipeline_status", "COMPLETED")
        logger.info("Modèle entraîné et sauvegardé avec succès")
        
        return saved_files

    except Exception as e:
        logger.error("Erreur lors de l'entraînement: %s", str(e))
        if mlflow.active_run():
            mlflow.set_tag("pipeline_status", "FAILED")
            mlflow.set_tag("error_message", str(e))
        raise

def evaluate_model():
    """
    Évaluation du modèle avec suivi MLflow.
    """
    try:
        # Configuration MLflow
        setup_mlflow()
        
        with mlflow.start_run(run_name="model_evaluation") as run:
            mlflow.set_tag("pipeline_type", "evaluation")
            
            # Chargement des données de test
            logger.info("Chargement des données de test")
            X_test = pd.read_csv(CLEAN_DATA_PATH / "X_test.csv")
            X_train = pd.read_csv(CLEAN_DATA_PATH / "X_train.csv")
            y_test = pd.read_csv(CLEAN_DATA_PATH / "y_test.csv")["RainTomorrow"]
            y_train = pd.read_csv(CLEAN_DATA_PATH / "y_train.csv")["RainTomorrow"]
            
            mlflow.log_param("evaluation.X_test_shape", str(X_test.shape))
            mlflow.log_param("evaluation.X_train_shape", str(X_train.shape))

            # Chargement du modèle et du scaler
            model = joblib.load(MODEL_PATH / "rfc.joblib")
            scaler = joblib.load(MODEL_PATH / "scaler.joblib")
            
            mlflow.log_param("evaluation.model_path", str(MODEL_PATH / "rfc.joblib"))
            mlflow.log_param("evaluation.scaler_path", str(MODEL_PATH / "scaler.joblib"))

            # Préparation des données
            X_test_scaled = scaler.transform(X_test)

            # Prédictions
            y_test_pred = model.predict(X_test)
            y_test_probs = model.predict_proba(X_test_scaled)[:,1]
            
            # Métriques sur les données de test
            test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
            test_precision = metrics.precision_score(y_test, y_test_pred)
            test_recall = metrics.recall_score(y_test, y_test_pred)
            test_f1 = metrics.f1_score(y_test, y_test_pred)
            test_roc_auc = metrics.roc_auc_score(y_test, y_test_probs)
            
            mlflow.log_metric("evaluation.test_accuracy", test_accuracy)
            mlflow.log_metric("evaluation.test_precision", test_precision)
            mlflow.log_metric("evaluation.test_recall", test_recall)
            mlflow.log_metric("evaluation.test_f1", test_f1)
            mlflow.log_metric("evaluation.test_roc_auc", test_roc_auc)
            
            # Matrice de confusion
            cm = metrics.confusion_matrix(y_test, y_test_pred)
            mlflow.log_param("evaluation.confusion_matrix", str(cm))

            # Prédiction sur les données d'entraînement
            y_train_pred = model.predict(X_train)
            y_train_probs = model.predict_proba(X_train)[:,1]
            
            # Métriques sur les données d'entraînement
            train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
            train_precision = metrics.precision_score(y_train, y_train_pred)
            train_recall = metrics.recall_score(y_train, y_train_pred)
            train_f1 = metrics.f1_score(y_train, y_train_pred)
            train_roc_auc = metrics.roc_auc_score(y_train, y_train_probs)
            
            mlflow.log_metric("evaluation.train_accuracy", train_accuracy)
            mlflow.log_metric("evaluation.train_precision", train_precision)
            mlflow.log_metric("evaluation.train_recall", train_recall)
            mlflow.log_metric("evaluation.train_f1", train_f1)
            mlflow.log_metric("evaluation.train_roc_auc", train_roc_auc)

            # Format original des métriques
            metrics_rfc = {
                "test": {
                    "accuracy": float(test_accuracy),
                    "precision": float(test_precision),
                    "recall": float(test_recall),
                    "f1": float(test_f1),
                    "roc_auc": float(test_roc_auc)
                },
                "train": {
                    "accuracy": float(train_accuracy),
                    "precision": float(train_precision),
                    "recall": float(train_recall),
                    "f1": float(train_f1),
                    "roc_auc": float(train_roc_auc)
                }
            }

            logger.info("Évaluation terminée")

            # Sauvegarde des métriques
            metrics_file = METRICS_DATA_PATH / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics_rfc, f, indent=4)
            
            mlflow.log_artifact(str(metrics_file))
            
            # Promotion du modèle basée sur les performances
            try:
                client = mlflow.tracking.MlflowClient()
                latest_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
                
                if latest_versions:
                    # Trier par version
                    latest_version = sorted(latest_versions, key=lambda x: int(x.version), reverse=True)[0]
                    
                    # Critère de promotion au stage "Staging"
                    if test_f1 > 0.7:  # Exemple de seuil
                        client.transition_model_version_stage(
                            name=MODEL_NAME,
                            version=latest_version.version,
                            stage="Staging"
                        )
                        mlflow.set_tag("model_promotion", "Promoted to Staging")
                    else:
                        mlflow.set_tag("model_promotion", "Performance below threshold")
            except Exception as e:
                logger.warning(f"Erreur lors de la promotion du modèle: {str(e)}")
                mlflow.set_tag("model_promotion_error", str(e))
            
            mlflow.set_tag("evaluation_status", "COMPLETED")
            return metrics_rfc

    except Exception as e:
        logger.error("Erreur lors de l'évaluation: %s", str(e))
        if mlflow.active_run():
            mlflow.set_tag("evaluation_status", "FAILED")
            mlflow.set_tag("error_message", str(e))
        raise

def predict_weather():
    """
    Prediction on new data with MLflow tracking.
    Checks if there's already an active MLflow run before creating a new one.
    """
    try:
        # MLflow setup
        setup_mlflow()
        
        # Check if there's already an active MLflow run
        active_run = mlflow.active_run()
        create_new_run = active_run is None
        
        # Function to execute the prediction
        def execute_prediction():
            # Try to load the model from MLflow
            try:
                # Find the most recent model in Production or Staging
                client = mlflow.tracking.MlflowClient()
                model_name = MODEL_NAME
                
                mlflow.log_param("prediction.model_name", model_name)
                
                # Find the most recent version in Production or Staging
                versions = client.search_model_versions(f"name='{model_name}'")
                production_versions = [mv for mv in versions if mv.current_stage in ["Production", "Staging"]]
                
                if production_versions:
                    # Sort by decreasing version
                    latest_prod = sorted(production_versions, key=lambda x: int(x.version), reverse=True)[0]
                    mlflow.set_tag("prediction.model_source", f"MLflow Registry - {latest_prod.current_stage}")
                    mlflow.set_tag("prediction.model_version", latest_prod.version)
                    
                    # Load the model from MLflow
                    model_uri = f"models:/{model_name}/{latest_prod.current_stage}"
                    model = mlflow.sklearn.load_model(model_uri)
                    
                    # Load the scaler separately
                    scaler = joblib.load(MODEL_PATH / "scaler.joblib")
                else:
                    raise ValueError("No production or staging model found in registry")
            
            except Exception as e:
                logger.warning(f"Failed to load from MLflow: {str(e)}. Using local files.")
                mlflow.set_tag("prediction.model_source", "Local files")
                mlflow.set_tag("prediction.load_from_mlflow_error", str(e))
                
                model = joblib.load(MODEL_PATH / "rfc.joblib")
                scaler = joblib.load(MODEL_PATH / "scaler.joblib")
            
            # Input file for prediction
            input_file = PREDICTION_RAW_DATA_PATH / csv_file_daily_prediction
            mlflow.log_param("prediction.input_file", str(input_file))
            logger.info(f"prediction input file: {input_file}")

            # Prepare the data
            input_df, _, _ = extract_and_prepare_df(
                PREDICTION_RAW_DATA_PATH, 
                csv_file_daily_prediction,
                log_to_mlflow=True  # No need to duplicate preparation logs here
            )
            
            mlflow.log_param("prediction.input_shape", str(input_df.shape))
        

            # Remove "RainTomorrow" column before prediction
            if "RainTomorrow" in input_df.columns:
                input_df = input_df.drop(columns=["RainTomorrow"])
            
            # Standardize the data
            input_scaled = scaler.transform(input_df)

            # Prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Log result
            mlflow.log_param("prediction.result", int(prediction))
            mlflow.log_param("prediction.probability", float(probability))
            mlflow.log_param("prediction.result_label", "Yes" if prediction == 1 else "No")
            
            mlflow.set_tag("prediction_status", "COMPLETED")
            logger.info("Prediction completed successfully")
            
            return prediction, float(probability)
        
        # Execute with the appropriate MLflow context
        if create_new_run:
            # If no run is active, create a new one
            with mlflow.start_run(run_name="weather_prediction") as run:
                mlflow.set_tag("pipeline_type", "prediction")
                result = execute_prediction()
        else:
            # Otherwise, use the existing run
            logger.info(f"Using existing MLflow run: {active_run.info.run_id}")
            result = execute_prediction()
            
        return result

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        if mlflow.active_run():
            mlflow.set_tag("prediction_status", "FAILED")
            mlflow.set_tag("error_message", str(e))
        raise