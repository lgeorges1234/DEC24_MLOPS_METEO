"""
MLflow configuration module for centralizing tracking settings
"""
import os
import mlflow
import logging

logger = logging.getLogger(__name__)

# MLflow server URI - can be overridden with environment variable
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
# MLflow registry URI - defaults to tracking URI if not specified
MLFLOW_REGISTRY_URI = os.environ.get("MLFLOW_REGISTRY_URI", MLFLOW_TRACKING_URI)

# Default experiment name
DEFAULT_EXPERIMENT_NAME = "weather_prediction"
# Model registry name
MODEL_NAME = "weather_prediction_model"

def setup_mlflow():
    """Configure MLflow tracking settings"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
        
        # Set registry URI
        mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
        logger.info(f"MLflow registry URI set to: {MLFLOW_REGISTRY_URI}")

        # Create the experiment if it doesn't exist
        experiment = mlflow.get_experiment_by_name(DEFAULT_EXPERIMENT_NAME)
        if experiment is None:
            mlflow.create_experiment(DEFAULT_EXPERIMENT_NAME)
            logger.info(f"Created new MLflow experiment: {DEFAULT_EXPERIMENT_NAME}")
        
        # Set the experiment as active
        mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)
        
        # Verify connection to MLflow server
        try:
            client = mlflow.tracking.MlflowClient()
            client.search_experiments()  # This will fail if connection is not working
            logger.info("Successfully connected to MLflow server")
        except Exception as e:
            logger.warning(f"Warning: Error connecting to MLflow server: {e}")
            logger.warning("Metrics and artifacts might not be properly logged")

        logger.info("MLflow setup completed successfully")
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        raise