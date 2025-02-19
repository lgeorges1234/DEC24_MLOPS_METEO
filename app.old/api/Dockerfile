FROM python:3.10-slim

WORKDIR /api

# Mise à jour de l'environnement python
# Installation de PostgreSQL (pour communiquer avec notre de données qui sera créée sur PosgreSQL)
# Suppression des fichiers temporaires pour réduire la taille de l'image
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Liste des fichiers necessaires au lancement de l'api
COPY requirements.txt .
COPY static/ ./static/
COPY utils/ ./utils/
COPY user_api.py .
COPY extract_api.py .
COPY training_api.py .
COPY evaluate_api.py .
COPY predict_api.py .
COPY init_db.py .
COPY main.py .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Gestion des authorisation d'execution du script de lancement

EXPOSE 8000

# Commande de lancement
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]