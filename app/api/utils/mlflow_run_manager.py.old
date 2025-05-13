"""
Utility for managing MLflow runs across multiple API endpoints
"""
import mlflow
import logging
from datetime import datetime
from utils.mlflow_config import setup_mlflow, DEFAULT_EXPERIMENT_NAME, MODEL_NAME

logger = logging.getLogger(__name__)

def start_workflow_run(workflow_name="weather_prediction_workflow"):
    """
    Starts a new workflow run that will span multiple endpoints.
    Returns the run_id that can be passed between endpoints.
    """
    # Make sure MLflow is configured
    setup_mlflow()
    
    # End any active run (shouldn't be needed but as a safety)
    if mlflow.active_run():
        mlflow.end_run()
    
    # Add timestamp to run name for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{workflow_name}_{timestamp}"
    
    # Start a new run
    run = mlflow.start_run(run_name=run_name)
    run_id = run.info.run_id
    
    # Log initial workflow information
    mlflow.set_tag("workflow_type", workflow_name)
    mlflow.set_tag("workflow_start_time", timestamp)
    mlflow.set_tag("workflow_status", "STARTED")
    
    # End the run for now (we'll resume it in other endpoints)
    mlflow.end_run()
    
    logger.info(f"Started new workflow run with ID: {run_id}")
    return run_id

def continue_workflow_run(run_id, step_name):
    """
    Continues an existing workflow run for a specific step.
    Returns the run context that should be used in a 'with' statement.
    Ensures any existing run is properly ended first.
    """
    # Make sure MLflow is configured
    setup_mlflow()
    
    # End any active run to avoid nested runs
    active_run = mlflow.active_run()
    if active_run and active_run.info.run_id != run_id:
        logger.info(f"Ending active run {active_run.info.run_id} to continue run {run_id}")
        mlflow.end_run()
    
    # Set the experiment to ensure we're in the right one
    mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)
    
    # Resume the existing run
    run = mlflow.start_run(run_id=run_id, nested=False)
    
    # Log step information
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mlflow.set_tag(f"step_{step_name}_start_time", timestamp)
    mlflow.set_tag("current_step", step_name)
    
    logger.info(f"Continuing workflow run {run_id} for step '{step_name}'")
    return run

def complete_workflow_run(run_id, status="COMPLETED", error_message=None):
    """
    Marks a workflow run as completed or failed.
    This should be called at the end of the workflow or when an error occurs.
    
    Args:
        run_id (str): The MLflow run ID to update
        status (str): Status to set (typically "COMPLETED" or "FAILED")
        error_message (str, optional): Error message to log if status is "FAILED"
    """
    # Make sure MLflow is configured
    setup_mlflow()
    
    # Resume the existing run
    with mlflow.start_run(run_id=run_id, nested=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mlflow.set_tag("workflow_end_time", timestamp)
        mlflow.set_tag("workflow_status", status)
        
        if error_message:
            mlflow.set_tag("workflow_error", error_message)
    
    logger.info(f"Completed workflow run {run_id} with status: {status}")

def get_deployment_run():
    """
    Get or create a deployment run for the current champion model version.
    Returns the run_id and model_version.
    """
    try:
        # Make sure MLflow is configured
        setup_mlflow()
        
        # Find the champion model
        client = mlflow.tracking.MlflowClient()
        
        try:
            # Get the champion model version
            champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
            model_version = champion_version.version
            logger.info(f"Found champion model version: {model_version}")
        except Exception as e:
            # If no champion alias exists, use the latest version
            logger.warning(f"No champion model found: {str(e)}. Using latest version.")
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            if not versions:
                raise ValueError(f"No versions found for model {MODEL_NAME}")
                
            latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
            model_version = latest_version.version
            logger.info(f"Using latest model version: {model_version}")
        
        # Get the experiment ID
        experiment = mlflow.get_experiment_by_name(DEFAULT_EXPERIMENT_NAME)
        if experiment is None:
            raise ValueError(f"Experiment {DEFAULT_EXPERIMENT_NAME} not found")
        
        # Check if we already have a deployment run for this model version
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.model_deployment = 'True' AND tags.model_version = '{model_version}'",
            max_results=1
        )
        
        if runs:
            # Found an existing deployment run
            deployment_run = runs[0]
            logger.info(f"Found existing deployment run: {deployment_run.info.run_id} for model version {model_version}")
            return deployment_run.info.run_id, model_version
        else:
            # Create a new deployment run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"model_deployment_v{model_version}_{timestamp}"
            
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.set_tag("model_deployment", "True")
                mlflow.set_tag("model_version", str(model_version))
                mlflow.set_tag("deployment_start_date", timestamp)
                mlflow.set_tag("pipeline_type", "deployment")
                
                logger.info(f"Created new deployment run: {run.info.run_id} for model version {model_version}")
                return run.info.run_id, model_version
    
    except Exception as e:
        logger.error(f"Error creating deployment run: {str(e)}")
        raise