from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime
from tasks.data_tasks import split_initial_dataset
from config import default_args, INITIAL_DATA_PATH


with DAG(
    'weather_initial_split_dag',
    default_args=default_args,
    description='Initial dataset splitting pipeline',
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2025, 2, 19),
    catchup=False,
    tags=['weather', 'data_split']
) as dag:
    
    check_initial_file = FileSensor(
        task_id='check_initial_file',
        filepath=str(INITIAL_DATA_PATH),
        fs_conn_id='fs_default',
        poke_interval=30,
        timeout=600,
        mode='poke'
    )
    
    split_data = PythonOperator(
        task_id='split_initial_dataset',
        python_callable=split_initial_dataset,
        provide_context=True
    )
    
    trigger_training = TriggerDagRunOperator(
        task_id='trigger_training_dag',
        trigger_dag_id='weather_training_dag',
        wait_for_completion=True
    )
    
    check_initial_file >> split_data >> trigger_training