app/
├── raw_data/
│   ├── initial_dataset/
│   │   └── weatherAUS.csv        # Original complete dataset
│   ├── training_raw_data/
│   │   └── weatherAUS_training.csv        # 2/3 of the data for training
│   └── prediction_raw_data/
│       └── weatherAUS_prediction.csv        # 1/3 of the data for predictions
│       └── daily_row_prediction.csv        # 1 line without target for daily prediction
│
├── api/
│   ├── data/
│   │   ├── prepared_data/
│   │   │   ├── meteo.csv
│   │   │   ├── X_train.csv
│   │   │   ├── y_train.csv
│   │   │   ├── X_test.csv
│   │   │   └── y_test.csv
│   │   ├── metrics/
│   │   │   └── metrics.json
│   │   └── models/
│   │       ├── rfc.joblib
│   │       └── scaler.joblib
│   │
│   ├── static/
│   ├── utils/
│   │   └── functions.py
│   ├── user_api.py
│   ├── extract_api.py
│   ├── training_api.py
│   ├── evaluate_api.py
│   ├── predict_api.py
│   ├── init_db.py
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── airflow/
│   ├── dags/
│   │   ├── tasks/
│   │   │   ├── __init__.py
│   │   │   ├── data_tasks.py
│   │   │   └── model_tasks.py
│   │   ├── config.py
│   │   ├── utils.py
│   │   ├── split_dag.py
│   │   ├── training_dag.py
│   │   └── prediction_dag.py
│   ├── logs/
│   └── plugins/
│
├── .env
└── docker-compose.yml