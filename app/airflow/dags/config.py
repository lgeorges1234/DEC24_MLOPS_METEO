from datetime import timedelta
import os
from pathlib import Path

# Shared configuration and paths
API_URL = os.getenv('API_API_URL', 'http://app:8000')
INITIAL_DATA_PATH = Path(os.getenv('INITIAL_DATA_PATH', '/app/raw_data/initial_dataset/weatherAUS.csv'))
TRAINING_RAW_DATA_PATH = Path(os.getenv('TRAINING_RAW_DATA_PATH', '/app/raw_data/training_raw_data'))
PREDICTION_RAW_DATA_PATH = Path(os.getenv('PREDICTION_RAW_DATA_PATH', '/app/raw_data/prediction_raw_data'))
CLEAN_DATA_PATH = Path(os.getenv('PREPARED_DATA_PATH', '/app/api/data/prepared_data'))
METRICS_DATA_PATH = Path(os.getenv('METRICS_DATA_PATH', '/app/api/data/metrics'))
MODEL_PATH = Path(os.getenv('MODEL_PATH', '/app/api/data/models'))

# Default DAG arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}