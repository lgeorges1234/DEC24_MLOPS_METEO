'''
Script de creation de la base de données utilisateurs
'''
#init_db.py
import os
import logging
import time
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import bcrypt

logger = logging.getLogger(__name__)

class DatabaseInitializer:
    '''
    Configuration de la base de données PostgreSQL
    '''
    def __init__(self):
        load_dotenv()
        self.db_params = {
            "host": os.getenv("DB_HOST"),
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD")
        }
        self._validate_env_vars()

    def get_connection_params(self):
        """
        Returns the database connection parameters.
        """
        return self.db_params

    def _validate_env_vars(self):
        missing = [k for k, v in self.db_params.items() if not v]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    def _wait_for_db(self, max_retries=5, delay=1):
        retries = max_retries
        while retries > 0:
            try:
                with psycopg2.connect(**self.db_params):
                    return True
            except psycopg2.OperationalError:
                retries -= 1
                logger.info("PostgreSQL non disponible, tentative restante: %s",retries)
                time.sleep(delay)
        return False

    def _create_enum_type(self, cursor):
        cursor.execute(
            """
            DO $$ 
            BEGIN 
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
                    CREATE TYPE user_role AS ENUM ('user', 'admin');
                END IF;
            END $$;
            """)

    def _create_users_table(self, cursor):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                email VARCHAR(100) NOT NULL UNIQUE,
                password_hashed VARCHAR(255) NOT NULL,
                role user_role NOT NULL DEFAULT 'user',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _create_admin_users(self, cursor):
        admins = [
            {
                "username": "admin_clemsim",
                "email": "admin_clemsim@example.com",
                "password": "SuperIPA2025!"
            }
        ]

        for admin in admins:
            hashed_password = bcrypt.hashpw(
                admin["password"].encode('utf-8'),
                bcrypt.gensalt()
            ).decode('utf-8')

            cursor.execute(
                """
                INSERT INTO users (username, email, password_hashed, role)
                VALUES (%s, %s, %s, 'admin')
                ON CONFLICT (username) DO NOTHING
                """,
                (admin["username"], admin["email"], hashed_password)
                )
            logger.info("Administrateurs créés")

    def init_db(self):
        '''
        Initialisation de la base de données utilisateurs
        '''
        if not self._wait_for_db():
            raise ConnectionError("PostgreSQL n'est pas disponible après plusieurs tentatives")

        try:
            with psycopg2.connect(**self.db_params) as conn:
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                with conn.cursor() as curs:
                    self._create_enum_type(curs)
                    self._create_users_table(curs)
                    self._create_admin_users(curs)
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error("Database initialization failed: %s", e)
            raise

def init_database():
    '''
    Initialisation de la base de données
    '''
    initializer = DatabaseInitializer()
    initializer.init_db()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_database()
