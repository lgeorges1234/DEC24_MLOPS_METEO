from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime
# Update these imports to reflect new path
from tasks.model_tasks import make_prediction
from config import default_args, PREDICTION_RAW_DATA_PATH, MODEL_PATH

with DAG(
    'weather_prediction_dag',
    default_args=default_args,
    description='Daily weather prediction pipeline',
    schedule_interval='0 0 * * *',
    start_date=datetime(2025, 2, 19),
    catchup=False,
    tags=['weather', 'prediction']
) as dag:
    
    check_prediction_file = FileSensor(
        task_id='check_prediction_file',
        filepath=str(PREDICTION_RAW_DATA_PATH / 'weatherAUS.csv'),
        fs_conn_id='fs_default',
        poke_interval=30,
        timeout=600,
        mode='poke'
    )
    
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

    [check_prediction_file, check_model_file] >> predict