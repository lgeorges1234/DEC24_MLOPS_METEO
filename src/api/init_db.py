import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

def init_database():
    # Chargement des variables d'environnement
    load_dotenv()
    
    # Récupérer les informations de connexion à partir du fichier de paramètrage .env
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    # Vérifier que toutes les variables d'environnement nécessaires sont présentes
    required_vars = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Variables d'environnement manquantes: {', '.join(missing_vars)}")
        sys.exit(1)

    try:
        # Se connecter à PostgreSQL
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        # Définir le niveau d'isolation pour permettre la création de type
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        # Créer un curseur
        curs = conn.cursor()
        
        print("Initialisation de la base de données...")

        # Créer le type ENUM user_role s'il n'existe pas
        curs.execute("""
            DO $$ 
            BEGIN 
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
                    CREATE TYPE user_role AS ENUM ('user', 'admin');
                END IF;
            END $$;
        """)
        print("Type ENUM user_role créé ou déjà existant")

        # Creation de la base utilisateur
        curs.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                email VARCHAR(100) NOT NULL UNIQUE,
                password_hashed VARCHAR(255) NOT NULL,
                role user_role NOT NULL DEFAULT 'user',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("Table users disponible")

        # Fermer le curseur
        curs.close()
        print("Base de données initialisée")

    except Exception as e:
        print(f"Erreur lors de l'initialisation de la base de données: {e}")
        sys.exit(1)
    finally:
        # Fermer la connexion
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    init_database()