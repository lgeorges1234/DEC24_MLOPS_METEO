"""
Utility for managing MLflow runs across multiple API endpoints
"""
import mlflow
import logging
from datetime import datetime
from utils.mlflow_config import setup_mlflow, DEFAULT_EXPERIMENT_NAME

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
    """
    # Make sure MLflow is configured
    setup_mlflow()
    
    # End any active run (shouldn't be needed but as a safety)
    if mlflow.active_run():
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
    This should be called at the end of the workflow.
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