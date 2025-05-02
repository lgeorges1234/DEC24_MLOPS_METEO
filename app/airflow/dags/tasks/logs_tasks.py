import logging
from airflow.operators.python import PythonOperator

# Configure logging if not already configured elsewhere
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_prediction_row_details(**context):
    """
    Task to log detailed information about the prediction row being used.
    This makes it easy to verify that the row changes with each DAG run.
    
    This function should be added to your DAG right after process_daily_prediction_row.
    """
    # Get the task instance
    ti = context['ti']
    
    # Try different task ID formats that might work
    task_id_options = [
        'data_preparation.process_daily_prediction_row',  # Format with dot
        'process_daily_prediction_row',                   # Just the task name
        'data_preparation__process_daily_prediction_row'  # Double underscore format
    ]
    
    row_processing_result = None
    
    # Try each format until we find one that works
    for task_id in task_id_options:
        try:
            result = ti.xcom_pull(task_ids=task_id)
            if result:
                logger.info(f"Successfully found XCom data using task ID: {task_id}")
                row_processing_result = result
                break
        except Exception as e:
            logger.info(f"Failed to get XCom with task ID {task_id}: {str(e)}")
    
    if not row_processing_result:
        logger.error("No result found from process_daily_prediction_row task. Check task ID format.")
        return
    
    # Extract the row details
    date = row_processing_result.get('date', 'Unknown')
    location = row_processing_result.get('location', 'Unknown')
    row_details = row_processing_result.get('row_details', {})
    
    # Format the information for clear display in logs
    logger.info("=" * 80)
    logger.info("DAILY PREDICTION ROW DETAILS:")
    logger.info("-" * 80)
    logger.info(f"Date: {date}")
    logger.info(f"Location: {location}")
    
    # If we have full row details, display them in a formatted way
    if row_details:
        logger.info("Full row details:")
        for key, value in row_details.items():
            logger.info(f"  {key}: {value}")
    
    # Include information about the prediction file
    daily_prediction_file = row_processing_result.get('daily_prediction_file', 'Unknown')
    logger.info(f"Daily prediction file: {daily_prediction_file}")
    
    # Include row count information
    rows_before = row_processing_result.get('rows_before', 'Unknown')
    rows_after = row_processing_result.get('rows_after', 'Unknown')
    logger.info(f"Rows in prediction dataset before: {rows_before}, after: {rows_after}")
    
    logger.info("=" * 80)
    
    # Return the information for potential downstream tasks
    return {
        "prediction_date": date,
        "prediction_location": location,
        "rows_remaining": rows_after
    }

