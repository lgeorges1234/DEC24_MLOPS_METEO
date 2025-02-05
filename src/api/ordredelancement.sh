#!/bin/bash
# Plusieurs scripts doivent être lancé, nous devons nous assurer de leur ordre de lancement

set -e

# Etape 1: Nous devons nous assurer que postgreSQL soit disponible avant de lancer le script d'initilisation de la base de données

until PGPASSWORD=$DB_PASSWORD psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c '\q'; do
  >&2 echo "PostgreSQL n'est pas disponibe"
  sleep 1
done

>&2 echo "PostgreSQL disponible, la base de données va être initaliser ainsi que l'api"

# Etape 2: Initialisation de la base de données
python init_db.py

# Etape 3: Demarrage de l'api
exec uvicorn user_api.py:app --host 0.0.0.0 --port 8000