'''
Point d'entrée principal de l'API FastAPI
'''
import logging
from fastapi import FastAPI
from endpoint.extract_api import router as extract_router
from endpoint.training_api import router as training_router
from endpoint.evaluate_api import router as evaluate_router
from endpoint.predict_api import router as predict_router
from endpoint.workflow_api import router as workflow_router

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Weather Prediction API")

# Include routers
app.include_router(extract_router, tags=["Data Extraction"])
app.include_router(training_router, tags=["Model Training"])
app.include_router(evaluate_router, tags=["Model Evaluation"])
app.include_router(predict_router, tags=["Inference"])
app.include_router(workflow_router, tags=["Workflows"])

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the Weather Prediction API",
        "endpoints": {
            "extract": "/extract",
            "training": "/training",
            "evaluate": "/evaluate",
                        "predict": {
                "automatic": "GET /predict",
                "manual": "POST /predict_user"
            },
            "workflow": {
                "start": "/workflow/start",
                "complete": "/workflow/complete",
                "run-full": "/workflow/run-full"
            }
        },
        "features_for_manual_input": {
            "Location": "Integer location code",
            "MinTemp": "Minimum temperature (-50 to 60)",
            "MaxTemp": "Maximum temperature (-50 to 60)",
            "WindGustDir": "Wind gust direction (0-360 degrees)",
            "WindGustSpeed": "Wind gust speed (0-200)",
            "WindDir9am": "Wind direction at 9am (0-360 degrees)",
            "WindDir3pm": "Wind direction at 3pm (0-360 degrees)",
            "WindSpeed9am": "Wind speed at 9am (0-200)",
            "WindSpeed3pm": "Wind speed at 3pm (0-200)",
            "Humidity9am": "Humidity at 9am (0-100%)",
            "Humidity3pm": "Humidity at 3pm (0-100%)",
            "Pressure3pm": "Atmospheric pressure (900-1100 hPa)",
            "Cloud9am": "Cloud cover at 9am (0-9)",
            "Cloud3pm": "Cloud cover at 3pm (0-9)",
            "RainToday": "Rain today (0 or 1)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    