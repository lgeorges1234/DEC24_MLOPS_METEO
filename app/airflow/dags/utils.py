import requests
from config import API_URL

def make_api_request(endpoint, timeout=300):
    """Generic function to make API requests"""
    try:
        response = requests.get(f"{API_URL}/{endpoint}", timeout=timeout)
        response.raise_for_status()
        result = response.json()
        print(f"Request to {endpoint} completed. Response: {result}")
        return result
    except requests.RequestException as e:
        print(f"Error making request to {endpoint} endpoint: {str(e)}")
        raise
    except Exception as e:
        print(f"Error in API request: {str(e)}")
        raise