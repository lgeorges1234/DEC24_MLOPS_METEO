'''
Lancement du main API
'''
#main.py
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from init_db import init_database
from user_api import router as users_router  # Importer le router user
from extract_api import router as extract_router# Importer le router d'exctraction et de préparation
from training_api import router as training_router # Importer le router training
from evaluate_api import router as evaluate_router # Importer le router d'évaluation
from predict_api import router as predict_router # Importer le router de prédiction

#Initialisation de la base de donnée PostreSQL
@asynccontextmanager
async def lifespan(_: FastAPI):
    '''
    Initialisation de la base de données lors du lancement de FastAPI
    '''
    init_database()
    yield

# Configuration de base
app = FastAPI(
    title = "Bienvenu sur l'API de prévision météorologique",
    lifespan = lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Ajout des différents routers
app.include_router(users_router, tags=["users"])
app.include_router(extract_router, tags=["extract"])
app.include_router(training_router, tags=["training"])
app.include_router(evaluate_router, tags=["evaluate"])
app.include_router(predict_router, tags=["predict"])

@app.get("/")
async def read_root():
    '''
    Message d'accueil
    '''
    return {"message": "Bienvenue sur l'API de prévision météorologique"}

#J'ai dù ajouter l'icone du navigateur pour éviter le message d'erreur lors du lancement du script
@app.get("/favicon.ico")
async def favicon():
    '''
    Icone du navigateur
    '''
    return FileResponse("static/favicon.ico")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    