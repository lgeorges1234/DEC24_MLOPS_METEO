import requests
import logging
import json
from config import API_URL

def make_api_request(endpoint, context, timeout=300):
    """Generic function to make API requests"""
    try:
        # Get run_id from XCom
        ti = context.get('ti')
        run_id = None

        if ti:
            workflow_response = ti.xcom_pull(task_ids='start_mlflow_workflow')
            
            # Extract run_id from the response if it's a dictionary
            if isinstance(workflow_response, dict):
                run_id = workflow_response.get('run_id')
            else:
                # Fall back to using the value directly if it's not a dict
                run_id = workflow_response
            
        # Build request parameters
        params = {"run_id": run_id} if run_id else {}

        # Make the request
        response = requests.get(f"{API_URL}/{endpoint}", params=params, timeout=timeout)
        response.raise_for_status()
        result = response.json()

        logging.info(f"{endpoint} API response: {json.dumps(result)}")
        return result

    except requests.RequestException as e:
        logging.error(f"Error making request to {endpoint} endpoint: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error in API request: {str(e)}")
        raise