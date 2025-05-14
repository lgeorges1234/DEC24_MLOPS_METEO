# Weather Prediction MLOps System

A comprehensive MLOps platform for daily rain prediction using Australian weather data, demonstrating a production-grade machine learning pipeline.

## Project Overview

This MLOps system is designed to predict whether it will rain tomorrow based on meteorological data. The project implements a complete machine learning lifecycle including data preparation, model training, automated retraining, and prediction serving through multiple interfaces.

## Architecture Diagram

![Weather Prediction MLOps Architecture](./reports/figures/Meteo_MLOPS_diagram.drawio.svg)

*The diagram above shows the complete architecture of the Weather Prediction MLOps system, including the data flow, services integration, and monitoring components.*

## Directory Structure

```
DEC24_MLOPS_METEO/
├── app/                         # Application code and Docker configuration
│   ├── airflow/                 # Airflow DAG definitions and configuration
│   ├── api/                     # FastAPI application for model serving
│   │   ├── data/                # Model artifacts and processed data
│   │   ├── endpoint/            # API endpoints implementation
│   │   ├── utils/               # Utility functions
│   │   ├── tests_unitaires/     # Unit tests
│   │   └── main.py              # API entry point
│   ├── raw_data/                # Directory for raw data files
│   │   ├── prediction_raw_data/ # Data for prediction
│   │   └── training_raw_data/   # Data for training
│   ├── streamlit_app/           # Streamlit web application
│   └── docker-compose.yaml      # Multi-container Docker configuration
├── initial_dataset/             # Initial weather dataset
│   └── weatherAUS.csv           # Australian weather data
├── notebooks/                   # Jupyter notebooks for exploration
└── reports/                     # Reports and visualizations
    └── figures/                 # Generated figures
```

## Key Components

### 1. Data Processing Pipeline

The system processes historical weather data from Australia to predict rain for the next day. The pipeline includes:

- Initial data splitting (2/3 for training, 1/3 for prediction simulation)
- Daily extraction of one row from prediction data to simulate real-time weather data
- Feature engineering and preprocessing with standardization and encoding

### 2. Orchestration via Airflow

Three main DAGs handle different aspects of the ML workflow:

- **Initial Split DAG**: Prepares datasets for simulation
- **Training DAG**: Weekly retraining of the model (every Monday at 00:00)
- **Prediction DAG**: Daily prediction using the current champion model (every day at 06:00)

### 3. Model Training and Serving

- **Random Forest Classifier**: Primary prediction model
- **MLflow Integration**: Tracking of model training, performance metrics, and model registry
- **Champion Model Selection**: Automatic promotion based on performance metrics
- **FastAPI Service**: Endpoints for model training and prediction

### 4. Web Interface

A Streamlit application provides a user-friendly interface for:

- Displaying weather predictions
- Visualizing model performance metrics
- Manual data entry for custom predictions

### 5. CI/CD Pipeline

- **GitHub Actions**: Automated testing and deployment
- **Docker Containers**: Isolated and reproducible environments for all services
- **Self-hosted Runner**: Direct deployment to the production environment

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Git
- Python 3.9+ (for local development)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DEC24_MLOPS_METEO.git
cd DEC24_MLOPS_METEO
```

2. Create a `.env` file in the `app` directory (use `.env.exemple` as a template)

3. Run the setup script to create necessary directories:
```bash
cd app
./setup_directories.sh
```

4. Start the services with Docker Compose:
```bash
docker-compose up -d
```

5. Initialize the data pipeline by triggering the first DAG:
```bash
curl -X POST http://localhost:8080/api/v1/dags/1_weather_initial_split_dag/dagRuns \
-H "Content-Type: application/json" \
-u "your_airflow_user:your_airflow_password" \
-d '{"conf": {}}'
```

### Accessing Services

- **Airflow**: http://localhost:8080
- **MLflow**: http://localhost:5000
- **FastAPI**: http://localhost:8000
- **Streamlit**: http://localhost:8501

## Model Details

The current implementation uses a Random Forest Classifier with the following configuration:

- 10 decision trees (`n_estimators: 10`)
- Maximum depth of 10 (`max_depth: 10`)
- Weekly retraining to adapt to new data
- Performance evaluation using accuracy, precision, recall, and F1-score
- Automatic promotion of models that improve accuracy

## Development Workflow

1. Create a new branch for your changes
2. Make changes and push to the remote repository
3. Create a Pull Request to the main branch
4. GitHub Actions will run tests to validate your changes
5. After review and approval, merge the PR
6. GitHub Actions will deploy the changes to the production environment

## Testing

Run the unit tests with:

```bash
cd app/api
pytest tests_unitaires/
```

Alternatively, use Docker for testing:

```bash
docker build -t weather-app-tests -f app/api/Dockerfile.test .
docker run --rm weather-app-tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Australian Bureau of Meteorology for the original data
- All contributors and maintainers of the project