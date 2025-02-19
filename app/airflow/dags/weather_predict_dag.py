from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import os
import requests
from pathlib import Path

# Environment variables and paths
API_URL = os.getenv('API_API_URL', 'http://app:8000')
RAW_DATA_PATH = Path(os.getenv('RAW_DATA_PATH', '/raw_data/weatherAUS.csv'))
CLEAN_DATA_PATH = Path(os.getenv('PREPARED_DATA_PATH', '/prepared_data'))
METRICS_DATA_PATH = Path(os.getenv('METRICS_DATA_PATH', '/metrics'))
MODEL_PATH = Path(os.getenv('MODEL_PATH', '/models'))

# Shared functions for API calls
def make_api_request(endpoint, timeout=300):
    """Generic function to make API requests"""
    try:
        response = requests.get(f"{API_URL}/{endpoint}", timeout=timeout)
        response.raise_for_status()
        result = response.json()
        print(f"Request to {endpoint} completed. Response: {result}")
        return result
    except requests.RequestException as e:
        print(f"Error making request to {endpoint} endpoint: {str(e)}")
        raise
    except Exception as e:
        print(f"Error in API request: {str(e)}")
        raise

# Training DAG functions
def collect_weather_data(**context):
    """Collect and preprocess weather data"""
    return make_api_request("extract")

def train_model(**context):
    """Train model using extracted and preprocessed data"""
    return make_api_request("training")

def evaluate_model(**context):
    """Evaluate trained model"""
    return make_api_request("evaluate")

# Prediction DAG function
def make_prediction(**context):
    """Make prediction using trained model"""
    return make_api_request("predict")

# DAG default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

# Training DAG - Runs weekly
with DAG(
    'weather_training_dag',
    default_args=default_args,
    description='Weekly weather model training pipeline',
    schedule_interval='0 0 * * MON',  # Every Monday at midnight
    start_date=datetime(2025, 2, 19),
    catchup=False,
    tags=['weather', 'training']
) as training_dag:
    
    # Check for raw data file
    check_raw_file = FileSensor(
        task_id='check_raw_data_file',
        filepath=str(RAW_DATA_PATH),
        fs_conn_id='fs_default',
        poke_interval=30,
        timeout=600,
        mode='poke'
    )

    with TaskGroup(group_id='data_collection') as data_collection:
        collect_weather = PythonOperator(
            task_id='collect_weather_data',
            python_callable=collect_weather_data,
            provide_context=True
        )

    # Check for cleaned data file
    check_clean_file = FileSensor(
        task_id='check_clean_data_file',
        filepath=str(CLEAN_DATA_PATH / "meteo.csv"),
        fs_conn_id='fs_default',
        poke_interval=30,
        timeout=600,
        mode='poke'
    )

    with TaskGroup(group_id='model_training') as model_training:
        train = PythonOperator(
            task_id='train_model',
            python_callable=train_model,
            provide_context=True
        )

    # Check for model files
    check_train_files = FileSensor(
        task_id='check_train_files',
        filepath=str(MODEL_PATH / "rfc.joblib"),
        fs_conn_id='fs_default',
        poke_interval=30,
        timeout=600,
        mode='poke'
    )

    with TaskGroup(group_id='model_evaluation') as model_evaluation:
        evaluate = PythonOperator(
            task_id='evaluate_model',
            python_callable=evaluate_model,
            provide_context=True
        )

    # Check for metrics file
    check_metrics_files = FileSensor(
        task_id='check_metrics_file',
        filepath=str(METRICS_DATA_PATH / "metrics.json"),
        fs_conn_id='fs_default',
        poke_interval=30,
        timeout=600,
        mode='poke'
    )

    # Trigger prediction DAG after training is complete
    trigger_prediction = TriggerDagRunOperator(
        task_id='trigger_prediction_dag',
        trigger_dag_id='weather_prediction_dag',
        wait_for_completion=True
    )

    # Set up task dependencies
    check_raw_file >> data_collection >> check_clean_file >> \
    model_training >> check_train_files >> \
    model_evaluation >> check_metrics_files >> trigger_prediction

# Prediction DAG - Runs daily
with DAG(
    'weather_prediction_dag',
    default_args=default_args,
    description='Daily weather prediction pipeline',
    schedule_interval='0 0 * * *',  # Every day at midnight
    start_date=datetime(2025, 2, 19),
    catchup=False,
    tags=['weather', 'prediction']
) as prediction_dag:
    
    # Check if model exists before prediction
    check_model_file = FileSensor(
        task_id='check_model_file',
        filepath=str(MODEL_PATH / "rfc.joblib"),
        fs_conn_id='fs_default',
        poke_interval=30,
        timeout=600,
        mode='poke'
    )

    predict = PythonOperator(
        task_id='make_prediction',
        python_callable=make_prediction,
        provide_context=True
    )

    check_model_file >> predict