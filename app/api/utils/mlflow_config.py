"""
MLflow configuration module for centralizing tracking settings
"""
import os
import mlflow
import logging

logger = logging.getLogger(__name__)

# MLflow server URI - can be overridden with environment variable
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_REGISTRY_URI = os.environ.get("MLFLOW_REGISTRY_URI", MLFLOW_TRACKING_URI)  # Default to tracking URI if not specified
MLFLOW_MAX_RETRIES=5
MLFLOW_RETRY_DELAY=3
MLFLOW_REQUEST_TIMEOUT=60

# Default experiment name
DEFAULT_EXPERIMENT_NAME = "weather_prediction"
# Model registry name
MODEL_NAME = "weather_prediction_model"

def setup_mlflow():
    """Configure MLflow tracking and registry settings"""
    try:
        # Set the tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
        
        # Set the registry URI
        mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
        logger.info(f"MLflow registry URI set to: {MLFLOW_REGISTRY_URI}")
        
        # Set HTTP request options via MLflow environment variables
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = str(MLFLOW_MAX_RETRIES)
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(MLFLOW_REQUEST_TIMEOUT)
        
        # Create the experiment if it doesn't exist
        experiment = mlflow.get_experiment_by_name(DEFAULT_EXPERIMENT_NAME)
        if experiment is None:
            mlflow.create_experiment(DEFAULT_EXPERIMENT_NAME)
            logger.info(f"Created new MLflow experiment: {DEFAULT_EXPERIMENT_NAME}")
        
        # Set the experiment as active
        mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)
        
        logger.info("MLflow setup completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        raise