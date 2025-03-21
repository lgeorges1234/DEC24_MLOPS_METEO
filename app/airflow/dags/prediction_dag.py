from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.task_group import TaskGroup
from datetime import datetime
from tasks.data_tasks import process_daily_prediction_row
from tasks.model_tasks import make_prediction
from config import (
    default_args, PREDICTION_RAW_DATA_PATH, 
    MODEL_PATH
)


with DAG(
    '3_weather_prediction_dag',
    default_args=default_args,
    description='Daily weather prediction pipeline',
    schedule_interval='0 6 * * *',  # Run daily at 6 AM
    start_date=datetime(2025, 2, 19),
    catchup=False,
    tags=['weather', 'prediction']
) as dag:
    
    check_prediction_file = FileSensor(
        task_id='check_prediction_file',
        filepath=str(PREDICTION_RAW_DATA_PATH / 'weatherAUS_prediction.csv'),
        fs_conn_id='fs_default',
        poke_interval=30,
        timeout=600,
        mode='poke'
    )

    with TaskGroup(group_id='data_preparation') as data_preparation:
        process_row = PythonOperator(
            task_id='process_daily_prediction_row',
            python_callable=process_daily_prediction_row,
            provide_context=True
        )

    check_daily_prediction_file = FileSensor(
        task_id='check_daily_prediction_file',
        filepath=str(PREDICTION_RAW_DATA_PATH / 'daily_row_prediction.csv'),
        fs_conn_id='fs_default',
        poke_interval=30,
        timeout=600,
        mode='poke'
    )

    # Check if model exists in MLflow registry or locally
    # check_model_availability = PythonSensor(
    #     task_id='check_model_availability',
    #     python_callable=check_model_in_registry,
    #     poke_interval=30,
    #     timeout=600,
    #     mode='poke'
    # )

    # Fallback file sensor in case the model is not in the registry
    check_model_file = FileSensor(
        task_id='check_model_file',
        filepath=str(MODEL_PATH / 'rfc.joblib'),
        fs_conn_id='fs_default',
        poke_interval=30,
        timeout=600,
        mode='poke'
    )

    with TaskGroup(group_id='prediction') as prediction_task_group:
        predict = PythonOperator(
            task_id='make_predictions',
            python_callable=make_prediction,
            provide_context=True,
        )

    # Set up task dependencies
    check_prediction_file >> data_preparation >> check_daily_prediction_file
    [check_daily_prediction_file, check_model_file] >> prediction_task_group
    # [check_daily_prediction_file, check_model_availability] >> prediction_task_group
    # check_model_availability.on_failure_trigger >> check_model_file >> prediction_task_group