'''
Liste des fonctions que nous appellerons dans les différents scripts
Avec intégration MLflow pour le tracking et le registre de modèles
Version optimisée avec tracking MLflow simplifié
'''

from pathlib import Path
from datetime import datetime
import time
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
import json
import os
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

from utils.mlflow_config import MLFLOW_MAX_RETRIES, MLFLOW_REGISTRY_URI, MLFLOW_RETRY_DELAY, setup_mlflow, MODEL_NAME

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
    """Equation outliers."""
    quartile1 = dataframe[column].quantile(q1)
    quartile3 = dataframe[column].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return up_limit, low_limit

def replace_with_thresholds(dataframe, column):
    """Remplace les outliers par les valeurs limites."""
    low_limit, up_limit = outlier_thresholds(dataframe, column)
    dataframe.loc[(dataframe[column] < low_limit), column] = low_limit
    dataframe.loc[(dataframe[column] > up_limit), column] = up_limit

def extract_and_prepare_df(path_raw_data, csv_file, user_input = None, log_to_mlflow=True):
    """
    Extraction et préparation des données météorologiques à partir du .csv
    Version avec logging MLflow simplifié
    """
    try:
        # Choix de la source de données
        if user_input is not None:
            # Conversion des données utilisateur en DataFrame
            df = pd.DataFrame([user_input])

            # Ajout de deux colonnes manquantes dans les données saisies mais présente dans les données d'entrainement
            if 'Evaporation' not in df.columns:
                df['Evaporation'] = 0.0 
            if 'Sunshine' not in df.columns:
                df['Sunshine'] = 0.0  

            initial_shape = df.shape
            logger.info("Données utilisateur chargées")
        else:
            # Extraction
            logger.info("Lecture du fichier: %s", path_raw_data / csv_file)
            if not (path_raw_data / csv_file).exists():
                raise FileNotFoundError(f"Le fichier {path_raw_data / csv_file} n'a pas été trouvé")

            df = pd.read_csv(path_raw_data / csv_file)
            logger.info("Données chargées")
        
            # Stockez la forme initiale mais ne la loggez qu'à la fin 
            # pour éviter les collisions lors de multiples appels
            initial_shape = df.shape
        
        # Conversion des variables catégorielles cibles en variables binaires
        if 'RainTomorrow' in df.columns:
            df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
        
        if 'RainToday' in df.columns:
            df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})

        # Identification et séparation des colonnes catégorielles et continues
        categorical, continuous = [],[]
        for col in df.columns:
            if df[col].dtype == 'object':
                categorical.append(col)
            else:
                continuous.append(col)

        # Gestion des valeurs vides
        missing_before = df.isnull().sum().sum()
        for col in categorical:
            df[col] = df[col].fillna(df[col].mode()[0])
        missing_after = df.isnull().sum().sum()

        # Encodage des variables catégorielles
        lencoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            lencoders[col] = LabelEncoder()
            df[col] = lencoders[col].fit_transform(df[col])

        # Suppression des lignes avec valeurs manquantes dans les variables cibles
        rows_before = len(df)
        if 'RainToday' in df.columns and 'RainTomorrow' in df.columns:
            df = df.dropna(subset=['RainToday', 'RainTomorrow'])
        rows_after = len(df)
        
        # Modification des seuils des variables cibles
        columns_for_outliers = [
            col for col in df.columns 
            if col not in ['RainTomorrow', 'RainToday', 'Date', 'Location']
        ]
        for column in columns_for_outliers:
            replace_with_thresholds(df, column)

        # Suppression des colonnes non nécessaires
        columns_to_drop = ['Date', 'Temp3pm', 'Pressure9am', 'Temp9am', 'Rainfall']
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)

        logger.info("Préparation terminée. Dimensions finales: %s", df.shape)
        
        # Log simplified data preparation metrics at the end
        if log_to_mlflow:
            mlflow.set_tag("data_preparation.initial_shape", str(initial_shape))
            mlflow.set_tag("data_preparation.final_shape", str(df.shape))
            mlflow.set_tag("data_preparation.missing_values_removed", missing_before - missing_after)
            mlflow.set_tag("data_preparation.rows_dropped", rows_before - rows_after)

        # Définition du nom du fichier nettoyé
        cleaned_file_name = f"{Path(csv_file).stem}_cleaned.csv"

        # Sauvegarde du fichier nettoyé
        df.to_csv(CLEAN_DATA_PATH / cleaned_file_name, index=False)
        logger.info("Données nettoyées sauvegardées")

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
    Version avec logging MLflow simplifié ciblant les métriques essentielles.
    """
    try:
        saved_files = {}
        
        # 1. Préparation des données
        mlflow.set_tag("current_step", "data_preparation")
        logger.info("Extraction et préparation des données")

        # Input file for prediction
        input_file = TRAINING_RAW_DATA_PATH / csv_file_training
        logger.info(f"training input file: {input_file}")

        df, lencoders, cleaned_file = extract_and_prepare_df(
            TRAINING_RAW_DATA_PATH, 
            csv_file_training,
            log_to_mlflow=True
        )
        
        # 2. Chargement des données nettoyées
        mlflow.set_tag("current_step", "data_loading")
        logger.info("Chargement des données nettoyées")
        
        # Séparation de la variable cible des features
        X = df.drop(columns=["RainTomorrow"]).astype("float")
        y = df["RainTomorrow"]

        # Sauvegarde du nom des colonnes
        features_names = X.columns
        features_names = X.columns.tolist()
        joblib.dump(features_names, MODEL_PATH / "feature_order.joblib")
        # Log essential features information
        mlflow.log_param("features_count", len(features_names))
        
        # Paramètres pour la division des données
        test_size = 0.2
        random_state = 42
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Log essential dataset metrics
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        
        # Class distribution (important for imbalanced datasets)
        train_class_dist = y_train.value_counts(normalize=True).to_dict()
        mlflow.log_metric("train_positive_class_ratio", train_class_dist.get(1, 0))
        
        # Standardisation des données
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Sauvegarde des données
        pd.DataFrame(X_train_scaled, columns=features_names).to_csv(CLEAN_DATA_PATH / "X_train.csv", index=False)
        pd.Series(y_train).to_csv(CLEAN_DATA_PATH / "y_train.csv", index=False)
        pd.DataFrame(X_test, columns=features_names).to_csv(CLEAN_DATA_PATH / "X_test.csv", index=False)
        pd.Series(y_test).to_csv(CLEAN_DATA_PATH / "y_test.csv", index=False)
        
        saved_files["train_data"] = {
            "X_train": str(CLEAN_DATA_PATH / "X_train.csv"),
            "y_train": str(CLEAN_DATA_PATH / "y_train.csv")
        }
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
        model_type = "RandomForest"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type}_{timestamp}"
        
        # Add model type to MLflow tracking
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("model_timestamp", timestamp)

        # Création et entraînement du modèle
        rfc = RandomForestClassifier(**params)
        rfc.fit(X_train_scaled, y_train)
        
        # Prédictions et évaluation
        train_preds = rfc.predict(X_train_scaled)
        X_test_scaled = scaler.transform(X_test)
        test_preds = rfc.predict(X_test_scaled)
        
        # Log essential training metrics
        train_accuracy = metrics.accuracy_score(y_train, train_preds)
        train_precision = metrics.precision_score(y_train, train_preds)
        train_recall = metrics.recall_score(y_train, train_preds)
        train_f1 = metrics.f1_score(y_train, train_preds)
        train_roc_auc = metrics.roc_auc_score(y_train, rfc.predict_proba(X_train_scaled)[:,1])

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("train_roc_auc", train_roc_auc)
        
        # Log essential testing metrics
        test_accuracy = metrics.accuracy_score(y_test, test_preds)
        test_precision = metrics.precision_score(y_test, test_preds)
        test_recall = metrics.recall_score(y_test, test_preds)
        test_f1 = metrics.f1_score(y_test, test_preds)
        test_roc_auc = metrics.roc_auc_score(y_test, rfc.predict_proba(X_test_scaled)[:,1])
        
        
        # Log only top 5 most important features
        feature_importances = [(feature, importance) for feature, importance in 
                               zip(features_names, rfc.feature_importances_)]
        top_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)[:5]
        
        for feature, importance in top_features:
            mlflow.log_metric(f"importance_{feature}", float(importance))
        
        # Create model performance summary for the description
        performance_summary = (
            f"F1: {test_f1:.4f}, "
            f"Accuracy: {test_accuracy:.4f}, "
            f"Precision: {test_precision:.4f}, "
            f"Recall: {test_recall:.4f}, "
            f"ROC-AUC: {test_roc_auc:.4f}"
        )
        
        # Create model version tags for metadata
        model_tags = {
            "model_type": model_type,
            "timestamp": timestamp,
            "estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "f1_score": f"{test_f1:.4f}",
            "accuracy": f"{test_accuracy:.4f}",
            "top_feature": top_features[0][0] if top_features else "None",
            "top_feature_importance": f"{top_features[0][1]:.4f}" if top_features else "0",
            "performance_summary": performance_summary,
            "training_samples": str(len(X_train)),
            "test_samples": str(len(X_test))
        }

        # Signature du modèle pour MLflow
        signature = infer_signature(X_train_scaled, rfc.predict(X_train_scaled))
        
        # Enregistrement du modèle dans MLflow
        model_info = mlflow.sklearn.log_model(
            sk_model=rfc,
            artifact_path="model",
            signature=signature,
            input_example=X_train_scaled[:5],
            registered_model_name=MODEL_NAME,
            metadata={"performance_summary": performance_summary}
        )
        
        # Add tags to the model version
        for tag_key, tag_value in model_tags.items():
            mlflow.set_tag(tag_key, tag_value)

        mlflow.set_tag("performance_summary", performance_summary)

        # Sauvegarde locale (pour compatibilité)
        model_path = MODEL_PATH / "rfc.joblib"
        scaler_path = MODEL_PATH / "scaler.joblib"
        
        joblib.dump(rfc, model_path)
        joblib.dump(scaler, scaler_path)
        
        saved_files["model"] = str(model_path)
        saved_files["scaler"] = str(scaler_path)
        
        # Sauvegarder les métriques dans un fichier JSON (pour compatibilité)
        metrics_dict = {
            "train": {
                "accuracy": float(train_accuracy),
                "f1": float(train_f1)
            },
            "test": {
                "accuracy": float(test_accuracy),
                "precision": float(test_precision),
                "recall": float(test_recall),
                "f1": float(test_f1),
                "roc_auc": float(test_roc_auc)
            },
            "top_features": dict(top_features)
        }
        
        # Before logging the artifact
        metrics_file = METRICS_DATA_PATH / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=4)

        # Debug the file
        file_exists = metrics_file.exists()
        file_readable = os.access(str(metrics_file), os.R_OK)
        file_size = metrics_file.stat().st_size if file_exists else 0

        logger.info(f"File status - Exists: {file_exists}, Readable: {file_readable}, Size: {file_size} bytes")
        logger.info(f"Absolute path: {metrics_file.absolute()}")

        # Try logging with error capturing
        try:
            mlflow.log_artifact(str(metrics_file.absolute()))
            logger.info(f"Successfully logged artifact: {metrics_file.name}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {str(e)}")
            # Print more details about the exception
            import traceback
            logger.error(traceback.format_exc())
        
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
    Utilise les alias pour désigner le modèle champion au lieu des stages dépréciés.
    """
    try:
        # We assume we're already in an MLflow run context
        mlflow.set_tag("current_step", "evaluation")
        
        # Load the test data
        logger.info("Chargement des données de test")
        
        X_test = pd.read_csv(CLEAN_DATA_PATH / "X_test.csv")
        y_test = pd.read_csv(CLEAN_DATA_PATH / "y_test.csv")["RainTomorrow"]
        
        # Load the model and scaler
        model = joblib.load(MODEL_PATH / "rfc.joblib")
        scaler = joblib.load(MODEL_PATH / "scaler.joblib")
        
        # Prepare the data
        X_test_scaled = scaler.transform(X_test)

        # Predictions
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate and log accuracy score for evaluation
        test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
        mlflow.log_metric("eval_accuracy", test_accuracy)
        
        # Save metrics
        metrics_file = METRICS_DATA_PATH / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({"accuracy_score": float(test_accuracy)}, f, indent=4)
        
        mlflow.log_artifact(str(metrics_file))
        
        # Model promotion logic based on accuracy using aliases
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Get the current model version
            all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            latest_version = sorted(all_versions, key=lambda x: int(x.version), reverse=True)[0]
            current_version = latest_version.version
            current_performance = test_accuracy  # Current model's accuracy score
            
            # Try to find if there's a model with the "champion" alias
            try:
                champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
                champion_run = client.get_run(champion_version.run_id)
                
                # Get champion model accuracy score (check both possible metric names)
                if "eval_accuracy" in champion_run.data.metrics:
                    champion_performance = champion_run.data.metrics["eval_accuracy"]
                elif "test_accuracy" in champion_run.data.metrics:
                    champion_performance = champion_run.data.metrics["test_accuracy"]
                else:
                    logger.warning("Champion model has no accuracy score metric")
                    champion_performance = 0
                
                # Promote if accuracy is greater than or equal to current champion model
                if current_performance >= champion_performance:
                    # Set the current model version as the new "champion"
                    client.set_registered_model_alias(MODEL_NAME, "champion", current_version)
                    
                    mlflow.set_tag("model_promotion", f"New model promoted to champion (Accuracy: {current_performance:.4f} vs {champion_performance:.4f})")
                    logger.info(f"Model version {current_version} promoted to champion with improved accuracy")
                else:
                    # No promotion needed
                    mlflow.set_tag("model_promotion", "Not promoted (no accuracy improvement)")
                    logger.info(f"Model version {current_version} not promoted (Accuracy: {current_performance:.4f} vs {champion_performance:.4f})")
            
            except Exception as e:
                # No champion model exists yet - always promote the first model
                client.set_registered_model_alias(MODEL_NAME, "champion", current_version)
                mlflow.set_tag("model_promotion", "First model assigned as champion")
                logger.info(f"Model version {current_version} assigned as first champion")
        
        except Exception as e:
            logger.error(f"Error during model promotion: {str(e)}")
            mlflow.set_tag("model_promotion_error", str(e))
        
        mlflow.set_tag("evaluation_status", "COMPLETED")
        return {"accuracy_score": float(test_accuracy)}
    
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation: {str(e)}")
        mlflow.set_tag("evaluation_status", "FAILED")
        mlflow.set_tag("evaluation_error", str(e))
        raise
        

def predict_weather(user_input = None):
    """
    Prediction on new data with MLflow tracking.
    Modified to use exclusively MLflow registry models.
    """
    try:
        # Tag the current step
        mlflow.set_tag("current_step", "prediction")
        
        # Set up MLflow with both tracking and registry URIs
        setup_mlflow()
        
        feature_order = joblib.load(MODEL_PATH / "feature_order.joblib")
        
        # Load model from MLflow with retry logic - no fallback
        client = mlflow.tracking.MlflowClient()
        model_name = MODEL_NAME
        
        # Get the model version using the champion alias with retry logic
        champion_version = None
        for attempt in range(MLFLOW_MAX_RETRIES):
            try:
                champion_version = client.get_model_version_by_alias(model_name, "champion")
                if champion_version:
                    logger.info(f"Found champion model version: {champion_version.version}")
                    break
                else:
                    logger.warning(f"No champion model found for {model_name}, attempt {attempt+1}/{MLFLOW_MAX_RETRIES}")
                    if attempt < MLFLOW_MAX_RETRIES - 1:
                        time.sleep(MLFLOW_RETRY_DELAY)
            except Exception as e:
                logger.warning(f"Error getting champion model version, attempt {attempt+1}/{MLFLOW_MAX_RETRIES}: {str(e)}")
                if attempt < MLFLOW_MAX_RETRIES - 1:
                    time.sleep(MLFLOW_RETRY_DELAY)
                else:
                    raise ValueError(f"Failed to get champion model version after {MLFLOW_MAX_RETRIES} attempts: {str(e)}")
        
        if not champion_version:
            raise ValueError(f"No champion model found for {model_name} after {MLFLOW_MAX_RETRIES} attempts")
        
        # Set tags
        mlflow.set_tag("model_source", f"MLflow Registry")
        mlflow.set_tag("model_version", champion_version.version)
        
        # Load the model with retry
        model = None
        for attempt in range(MLFLOW_MAX_RETRIES):
            try:
                logger.info(f"Loading model from MLflow registry (attempt {attempt+1}/{MLFLOW_MAX_RETRIES})")
                model_uri = f"models:/{model_name}@champion"
                model = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Successfully loaded model from MLflow registry")
                break
            except Exception as e:
                logger.warning(f"Error loading model, attempt {attempt+1}/{MLFLOW_MAX_RETRIES}: {str(e)}")
                if attempt < MLFLOW_MAX_RETRIES - 1:
                    time.sleep(MLFLOW_RETRY_DELAY)
                else:
                    raise ValueError(f"Failed to load model after {MLFLOW_MAX_RETRIES} attempts: {str(e)}")
        
        # Load the scaler - still using local file for now
        scaler = joblib.load(MODEL_PATH / "scaler.joblib")
        
        # # Determine input source
        # if user_input is not None:
        #     logger.info("Preparing user input data")
        #     # User input prediction
        #     mlflow.set_tag("prediction_type", "user_input")
        #     input_df, _, _ = extract_and_prepare_df(
        #         PREDICTION_RAW_DATA_PATH, 
        #         csv_file_daily_prediction,
        #         user_input=user_input,
        #         log_to_mlflow=True
        #     )
        #     input_df = input_df.reindex(columns=feature_order)
        #     logger.info(f"Prepared input DataFrame: {input_df.shape}")
        # else:
        #     # Input file for prediction
        #     mlflow.set_tag("prediction_type", "file_based")
        #     input_file = PREDICTION_RAW_DATA_PATH / csv_file_daily_prediction
        #     mlflow.set_tag("prediction_input_file", str(input_file))
            
        #     # Log file details
        #     file_size = os.path.getsize(input_file) if os.path.exists(input_file) else 0
        #     file_mtime = datetime.fromtimestamp(os.path.getmtime(input_file)).strftime('%Y-%m-%d %H:%M:%S') if os.path.exists(input_file) else "unknown"
        #     logger.info(f"Using prediction file: {input_file} (size: {file_size}, modified: {file_mtime})")

        #     # Prepare the data
        #     input_df, _, _ = extract_and_prepare_df(
        #         PREDICTION_RAW_DATA_PATH, 
        #         csv_file_daily_prediction,
        #         log_to_mlflow=False
        #     )
        #     logger.info(f"Prepared input DataFrame: {input_df.shape}")

        logger.info("Preparing user input data")
        # User input prediction
        mlflow.set_tag("prediction_type", "user_input")
        input_df, _, _ = extract_and_prepare_df(
            PREDICTION_RAW_DATA_PATH, 
            csv_file_daily_prediction,
            user_input=user_input,
            log_to_mlflow=True
        )
        input_df = input_df.reindex(columns=feature_order)
        logger.info(f"Prepared input DataFrame: {input_df.shape}")

        # Remove "RainTomorrow" column before prediction
        if "RainTomorrow" in input_df.columns:
            input_df = input_df.drop(columns=["RainTomorrow"])

        # Standardize the data
        input_scaled = scaler.transform(input_df)

        # Make prediction using scaled data for BOTH prediction and probability
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Log detailed prediction information
        logger.info(f"Raw prediction: {prediction}, probability: {probability}")
        
        # Log prediction results
        mlflow.log_metric("prediction_result", int(prediction))
        mlflow.log_metric("prediction_probability", float(probability))
        mlflow.set_tag("prediction_label", "Yes" if prediction == 1 else "No")
        
        mlflow.set_tag("prediction_status", "COMPLETED")
        logger.info("Prediction completed successfully")
        
        return prediction, float(probability)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        mlflow.set_tag("prediction_status", "FAILED")
        mlflow.set_tag("error_message", str(e))
        raise