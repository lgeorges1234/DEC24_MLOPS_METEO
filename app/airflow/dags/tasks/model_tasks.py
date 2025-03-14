from utils import make_api_request

def collect_weather_data(**context):
    """Collect and preprocess weather data"""
    return make_api_request("extract", context)

def train_model(**context):
    """Train model using extracted and preprocessed data"""
    return make_api_request("training", context)

def evaluate_model(**context):
    """Evaluate trained model"""
    return make_api_request("evaluate", context)

def make_prediction(**context):
    """Make prediction using trained model"""
    return make_api_request("predict", context)