import requests
import logging
import json
from utils import make_api_request
from config import(
    API_URL
)


# Function to start a new workflow and return the run_id
def start_mlflow_workflow(**context):
    """Start a new MLflow workflow and store the run_id in XCom"""
    try:
        # response = requests.post(f"{API_URL}/workflow/start")
        result = make_api_request("workflow/start", context)

        run_id = result.get('run_id')
        
        if not run_id:
            raise ValueError("No run_id returned from workflow start")
        
        logging.info(f"Started MLflow workflow with run_id: {run_id}")
        return result
    
    except Exception as e:
        logging.error(f"Failed to start MLflow workflow: {str(e)}")
        raise

def complete_mlflow_workflow(**context):
    """Complete the MLflow workflow"""
    try:
        # Get run_id from XCom
        run_id = context['ti'].xcom_pull(task_ids='start_mlflow_workflow')
        
        # Complete workflow
        response = requests.post(
            f"{API_URL}/workflow/complete", 
            params={"run_id": run_id, "status": "COMPLETED"}
        )
        response.raise_for_status()
        
        result = response.json()
        logging.info(f"Workflow completion response: {json.dumps(result)}")
        
        return result
    except Exception as e:
        logging.error(f"Failed to complete workflow: {str(e)}")
        raise

# Alternative: Use the full workflow endpoint
def run_full_workflow(**context):
    """Run the complete workflow in one call"""
    try:
        response = requests.post(f"{API_URL}/workflow/run-full")
        response.raise_for_status()
        
        result = response.json()
        logging.info(f"Full workflow response: {json.dumps(result)}")
        
        # Store run_id in XCom for potential later use
        run_id = result.get('run_id')
        if run_id:
            context['ti'].xcom_push(key='run_id', value=run_id)
        
        return result
    except Exception as e:
        logging.error(f"Failed to run full workflow: {str(e)}")
        raise

# Function to call extract endpoint with the workflow run_id
# def call_extract_endpoint(**context):
#     """Call the extract endpoint with the workflow run_id"""
#     try:
#         # Get run_id from XCom
#         run_id = context['ti'].xcom_pull(task_ids='start_mlflow_workflow')
        
#         # Call extract API
#         response = requests.get(f"{API_URL}/extract", params={"run_id": run_id})
#         response.raise_for_status()
        
#         result = response.json()
#         logging.info(f"Extract API response: {json.dumps(result)}")
        
#         return result.get('output_file')
#     except Exception as e:
#         logging.error(f"Failed to call extract endpoint: {str(e)}")
#         raise
