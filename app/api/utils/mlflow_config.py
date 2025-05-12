"""
MLflow configuration module for centralizing tracking settings
"""
import os
import logging
import sys  # Importé pour vérifier 'pytest' dans sys.modules

logger = logging.getLogger(__name__)

# MLflow server URI - can be overridden with environment variable
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_REGISTRY_URI = os.environ.get("MLFLOW_REGISTRY_URI", MLFLOW_TRACKING_URI)  # Default to tracking URI if not specified
MLFLOW_MAX_RETRIES = 5
MLFLOW_RETRY_DELAY = 3
MLFLOW_REQUEST_TIMEOUT = 60

# Default experiment name
DEFAULT_EXPERIMENT_NAME = "weather_prediction"
# Model registry name
MODEL_NAME = "weather_prediction_model"

def in_test_mode():
    """
    Determines if the code is running in a test environment.
    """
    # Check environment variable
    if os.environ.get("TESTING", "").lower() in ("true", "1", "yes"):
        return True
    
    # Check for pytest in sys.modules
    if 'pytest' in sys.modules:
        return True
    
    # Check if the MLFLOW_TRACKING_URI contains 'fake' or 'mock'
    if "fake" in MLFLOW_TRACKING_URI or "mock" in MLFLOW_TRACKING_URI:
        return True
    
    return False

def setup_mlflow():
    """Configure MLflow tracking and registry settings"""
    # If in test mode, skip MLflow initialization
    if in_test_mode():
        logger.info("Test environment detected, skipping MLflow setup")
        return True
    
    try:
        # Import MLflow here to avoid errors if it's not available
        import mlflow
        
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
        # In test mode, continue despite errors
        if in_test_mode():
            logger.warning("Continuing despite MLflow setup error (test environment)")
            return True
        raise