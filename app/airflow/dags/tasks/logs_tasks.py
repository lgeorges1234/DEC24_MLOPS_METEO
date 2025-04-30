
from airflow.operators.python import PythonOperator

def log_prediction_row_details(**context):
    """
    Task to log detailed information about the prediction row being used.
    This makes it easy to verify that the row changes with each DAG run.
    
    This function should be added to your DAG right after process_daily_prediction_row.
    """
    import logging
    import json
    
    # Get the task instance
    ti = context['ti']
    
    # Pull the result from the process_daily_prediction_row task
    row_processing_result = ti.xcom_pull(task_ids='data_preparation.process_daily_prediction_row')
    
    if not row_processing_result:
        logging.error("No result found from process_daily_prediction_row task")
        return
    
    # Extract the row details
    date = row_processing_result.get('date', 'Unknown')
    location = row_processing_result.get('location', 'Unknown')
    row_details = row_processing_result.get('row_details', {})
    
    # Format the information for clear display in logs
    logging.info("=" * 80)
    logging.info("DAILY PREDICTION ROW DETAILS:")
    logging.info("-" * 80)
    logging.info(f"Date: {date}")
    logging.info(f"Location: {location}")
    
    # If we have full row details, display them in a formatted way
    if row_details:
        logging.info("Full row details:")
        for key, value in row_details.items():
            logging.info(f"  {key}: {value}")
    
    # Include information about the prediction file
    daily_prediction_file = row_processing_result.get('daily_prediction_file', 'Unknown')
    logging.info(f"Daily prediction file: {daily_prediction_file}")
    
    # Include row count information
    rows_before = row_processing_result.get('rows_before', 'Unknown')
    rows_after = row_processing_result.get('rows_after', 'Unknown')
    logging.info(f"Rows in prediction dataset before: {rows_before}, after: {rows_after}")
    
    logging.info("=" * 80)
    
    # Return the information for potential downstream tasks
    return {
        "prediction_date": date,
        "prediction_location": location,
        "rows_remaining": rows_after
    }