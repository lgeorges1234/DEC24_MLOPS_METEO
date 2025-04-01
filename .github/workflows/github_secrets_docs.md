# GitHub Secrets for Weather App Deployment

To successfully deploy the Weather App using GitHub Actions, you need to set up the following secrets in your GitHub repository:

## Repository Secrets

1. **`AIRFLOW_PASSWORD`** - Password for the Airflow webserver admin account

2. **`DB_NAME`** - PostgreSQL database name (used by both the app and MLflow)

3. **`DB_USER`** - PostgreSQL database username

4. **`DB_PASSWORD`** - PostgreSQL database password

5. **`DB_SECRET_KEY`** - Secret key for the application

6. **`SMTP_PASSWORD`** - Password for the SMTP server (used by Airflow for notifications)

7. **`SSH_PEM_KEY`** - The entire contents of a .pem key file for connecting to your EC2 instance

8. **`ACTIONS_RUNNER_TOKEN`** - GitHub token with permissions to register runners (for runner setup workflow)

## How to Add Secrets

1. Go to your GitHub repository
2. Click on **Settings**
3. Click on **Secrets and variables** → **Actions**
4. Click on **New repository secret**
5. Add each of the secrets listed above

## Server Requirements

Your deployment server should have:

- Docker and Docker Compose installed
- At least 4GB RAM
- At least 30GB disk space
- Open ports for:
  - 8080 (Airflow webserver)
  - 5000 (MLflow)
  - 8000 (Weather App API)
  - 5432 (PostgreSQL - can be internal only)
  - 6379 (Redis - can be internal only)