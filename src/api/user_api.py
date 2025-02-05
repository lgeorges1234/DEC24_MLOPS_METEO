# API utilisateur
from datetime import datetime
from typing import List
import os
from enum import Enum
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import jwt
import bcrypt
from pydantic import BaseModel, EmailStr
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

# Configuration de base
app = FastAPI(title="Bienvenu sur l'API de prévision meteorologique")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur l'API de prévision météorologique"}

#J'ai du ajouté cela 
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

# Dénifition des class pour la creation, le rôle et l'authentification de l'utilisateur 
class UserRole(str, Enum): # Enum permet de limiter le type de données d'entrées à celles définies dans la Class 
    USER = "user"
    ADMIN = "admin"

#Cette class reprend les informations communes aux autres classes d'utilisateurs 
class UserBase(BaseModel):
    email: EmailStr
    username: str

class UserCreate(UserBase):
    password: str
    role: UserRole = UserRole.USER

class User(UserBase):
    id: int
    role: UserRole
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

# Charger les variables d'environnement depuis le fichier .env contenant la configuration de notre db_user sous PostgreSQL
load_dotenv()

async def get_db_user():
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
''' 
#arguments utilisés
token: récupération du token via une authorisation 'OAuth2'. l'indication 'str' permet de rendre plus lisible le code en indiquant le type de données attendu
connection: se connecte à la base de données db_user pour récupérer les informations 
'''
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
'''
/token pour gérer les demandes d'authentification

'''
@app.post("/token", response_model=Token)
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
        
        if not user:                            ## Vérification du nom de l'utilisateur saisi dans le formulaire 
            raise auth_exception
            
        if not bcrypt.checkpw(                  ## bcrypt.checkpw() compare le mot de passe saisi avec celui de la base utilisateur
            form_data.password.encode('utf-8'),
            user['password_hashed'].encode('utf-8')
        ):
            raise auth_exception
        
        # Ajout de plus d'informations dans le token
        token_data = {
            "sub": str(user['id']), ## IMPORTANT: ne fonctionnait pas sans l'application de str() pour obtenir avoir une donnée serialisable.
            "role": user['role'],
            "email": user['email']
        }
        
        print(f"Creating token with data: {token_data}")  # check pour debogage

        access_token = jwt.encode(              ## Creation du token d'accès pour l'utilisateur à partir de son id ert rôle
            token_data,
            os.getenv("DB_SECRET_KEY"),
            algorithm="HS256"
        )
        return {"access_token": access_token, "token_type": "bearer"}

'''
Creation d'un nouvel utilisateur (post) en appelant la class UserCreate et en se connectant à la base de données utilisateurs
'''

@app.post("/users", response_model=User)
async def create_user(user: UserCreate, conn = Depends(get_db_user)):

    hashed_password = bcrypt.hashpw(             ## Hashage du mot de passe utilisateur pour la sécurisation 
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
            curs.execute(sql_statment, (user.username, user.email, hashed_password, user.role))
            conn.commit()                        ## Sauvegarde des informations du nouvel utilisateur dans notre base de donnée
            new_user = curs.fetchone()
            return User(**new_user)
        except psycopg2.Error as e:
            conn.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            ) from e

'''
endpoint /users - Accès aux données utilisateurs
'''
@app.get("/users", response_model=List[User])
async def get_active_user(active_user: User = Depends(get_user),conn = Depends(get_db_user)):

    print(f"Current user role: {active_user.role}")  # Check pour débogage
    if active_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous ne disposez pas des permissions administrateurs"
        )
    
    with conn.cursor(cursor_factory = RealDictCursor) as curs:
        sql_statment = "SELECT * FROM users ORDER BY created_at DESC"
        curs.execute(sql_statment)
        users = curs.fetchall()                  ##Récupération de toute la base d'utilisateurs
        return [User(**user) for user in users]  ##On retourne une liste d'utilisateur (users étant composés de dictionnaires)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)