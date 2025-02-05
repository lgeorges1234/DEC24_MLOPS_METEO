'''
Script de creation des endpoints utilisateurs 
'''
# user_api.py

from datetime import datetime
from typing import List
import os
from enum import Enum
from fastapi import Depends, APIRouter, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
import bcrypt
from pydantic import BaseModel, EmailStr
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

# Ajout d'un router pour pouvoir centraliser tous nos scripts de /endpoint dans le main.py

router = APIRouter()

# Configuration de l'authorisation

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dénifition des class pour la creation, le rôle et l'authentification de l'utilisateur
class UserRole(str, Enum):
    '''
    Configuration des rôles des utilisateurs
    '''
    # Enum permet de limiter le type de données d'entrées à celles définies dans la Class
    USER = "user"
    ADMIN = "admin"

#Cette class reprend les informations communes aux autres classes d'utilisateurs 
class UserBase(BaseModel):
    '''
    Configuration partagée par tous les utilisateurs
    '''
    email: EmailStr
    username: str

class UserCreate(UserBase):
    '''
    Configuration spécifique de l'utilisateur lors de sa creation
    '''
    password: str

class User(UserBase):
    '''
    Définition d'un utilisateur
    '''
    id: int
    role: UserRole
    created_at: datetime

    class Config:
        '''
        Configuration de l'utilisateur
        '''
        from_attributes = True

class Token(BaseModel):
    '''
    Définition du token d'accès
    '''
    access_token: str
    token_type: str

# Charger les variables d'environnement depuis le fichier .env
# Il contient la configuration de notre db_user sous PostgreSQL
load_dotenv()

async def get_db_user():
    '''
    Accès aux configurations de la base de données utilisateurs
    '''
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    try:
        yield conn
    finally:
        conn.close()

#Fonction pour interroger la base données utilisateurs en fonction du token

async def get_user(token: str = Depends(oauth2_scheme), conn = Depends(get_db_user)):
    '''
    Définition du message à renvoyer pour une erreur de connection
    '''
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Connexion non authorisée merci de vérifier vos droits",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, os.getenv("DB_SECRET_KEY"), algorithms=["HS256"])
        print(f"Payload décodé : {payload}")  # Affiche les données décodées du token
        user_id: int = payload.get("sub") ## Reconversion en entier
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError as exc:
        raise credentials_exception from exc

    with conn.cursor(cursor_factory = RealDictCursor) as curs:
        # https://www.geeksforgeeks.org/python-psycopg-cursor-class/
        sql_statment = "SELECT * FROM users WHERE id = %s"
        curs.execute(sql_statment, (user_id,))
        user = curs.fetchone()
        if user is None:
            raise credentials_exception
        return User(**user)

# Endpoints

@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), conn = Depends(get_db_user)):

    '''
    Extraction des données de l'utilisateur à partir du formulaire d'entrée via 'OAuth'
    Définition d'un message d'erreur lors du login
    '''
    auth_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="username ou mot de passe incorrect",
        headers={"WWW-Authenticate": "Bearer"},
    )

    with conn.cursor(cursor_factory = RealDictCursor) as curs:
        sql_statment = "SELECT * FROM users WHERE email = %s"
        curs.execute(
            sql_statment,
            (form_data.username,)
        )
        user = curs.fetchone()

        ## Vérification du nom de l'utilisateur saisi dans le formulaire
        if not user:
            raise auth_exception

        ## bcrypt.checkpw() compare le mot de passe saisi avec celui de la base utilisateur
        if not bcrypt.checkpw(
            form_data.password.encode('utf-8'),
            user['password_hashed'].encode('utf-8')
        ):
            raise auth_exception

        # Ajout de plus d'informations dans le token
        ## IMPORTANT:  str() pour obtenir avoir une donnée serialisable.
        token_data = {
            "sub": str(user['id']),
            "role": user['role'],
            "email": user['email']
        }

        print(f"Creating token with data: {token_data}")  # check pour debogage

        ## Creation du token d'accès pour l'utilisateur à partir de son id ert rôle
        access_token = jwt.encode(
            token_data,
            os.getenv("DB_SECRET_KEY"),
            algorithm="HS256"
        )
        return {"access_token": access_token, "token_type": "bearer"}

@router.post("/users", response_model=User)
async def create_user(user: UserCreate, conn = Depends(get_db_user)):
    '''
    Creation de l'utilisateur
    '''
    ## Hashage du mot de passe utilisateur pour la sécurisation
    hashed_password = bcrypt.hashpw(
        user.password.encode('utf-8'),
        bcrypt.gensalt()
    ).decode('utf-8')

    with conn.cursor(cursor_factory = RealDictCursor) as curs:    ## Creation du nouvel utilisateur
        sql_statment = '''
                INSERT INTO users (username, email, password_hashed, role)
                VALUES (%s, %s, %s, %s)
                RETURNING id, username, email, role, created_at;
            '''
        try:
            curs.execute(sql_statment, (user.username, user.email, hashed_password, UserRole.USER))
            ## Sauvegarde des informations du nouvel utilisateur dans notre base de donnée
            conn.commit()
            new_user = curs.fetchone()
            return User(**new_user)
        except psycopg2.Error as e:
            conn.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            ) from e

@router.get("/users", response_model=List[User])
async def get_active_user(active_user: User = Depends(get_user),conn = Depends(get_db_user)):

    '''
    Acceder à la base de données utilisateurs (pour les administrateurs)
    '''

    print(f"Current user role: {active_user.role}")  # Check pour débogage
    if active_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous ne disposez pas des permissions administrateurs"
        )

    with conn.cursor(cursor_factory = RealDictCursor) as curs:
        sql_statment = "SELECT * FROM users ORDER BY created_at DESC"
        curs.execute(sql_statment)
        ##Récupération de toute la base d'utilisateurs
        users = curs.fetchall()
        ##On retourne une liste d'utilisateur (users étant composés de dictionnaires)
        return [User(**user) for user in users]
    