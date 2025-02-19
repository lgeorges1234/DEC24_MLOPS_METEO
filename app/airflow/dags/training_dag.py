from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.task_group import TaskGroup
from datetime import datetime
from tasks.model_tasks import collect_weather_data, train_model, evaluate_model
from config import (
    default_args, TRAINING_RAW_DATA_PATH, 
    CLEAN_DATA_PATH, MODEL_PATH, METRICS_DATA_PATH
)


with DAG(
    'weather_training_dag',
    default_args=default_args,
    description='Weekly weather model training pipeline',
    schedule_interval='0 0 * * MON',
    start_date=datetime(2025, 2, 19),
    catchup=False,
    tags=['weather', 'training']
) as dag:
    
    check_raw_file = FileSensor(
        task_id='check_raw_data_file',
        filepath=str(TRAINING_RAW_DATA_PATH / 'weatherAUS.csv'),
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

    check_metrics_files = FileSensor(
        task_id='check_metrics_file',
        filepath=str(METRICS_DATA_PATH / "metrics.json"),
        fs_conn_id='fs_default',
        poke_interval=30,
        timeout=600,
        mode='poke'
    )

    trigger_prediction = TriggerDagRunOperator(
        task_id='trigger_prediction_dag',
        trigger_dag_id='weather_prediction_dag',
        wait_for_completion=True
    )

    # Set up task dependencies
    check_raw_file >> data_collection >> check_clean_file >> \
    model_training >> check_train_files >> \
    model_evaluation >> check_metrics_files >> trigger_prediction
