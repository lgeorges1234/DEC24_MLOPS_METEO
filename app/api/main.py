'''
Point d'entrée principal de l'API FastAPI
'''
import logging
from fastapi import FastAPI
from endpoint.extract_api import router as extract_router
from endpoint.training_api import router as training_router
from endpoint.evaluate_api import router as evaluate_router
from endpoint.workflow_api import router as workflow_router

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Weather Prediction API")

# Include routers
app.include_router(extract_router, tags=["Data Extraction"])
app.include_router(training_router, tags=["Model Training"])
app.include_router(evaluate_router, tags=["Model Evaluation"])
app.include_router(workflow_router, tags=["Workflows"])

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the Weather Prediction API",
        "endpoints": {
            "extract": "/extract",
            "training": "/training",
            "evaluate": "/evaluate",
            "workflow": {
                "start": "/workflow/start",
                "complete": "/workflow/complete",
                "run-full": "/workflow/run-full"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    