import pandas as pd
from config import INITIAL_DATA_PATH, TRAINING_RAW_DATA_PATH, PREDICTION_RAW_DATA_PATH

def split_initial_dataset(**context):
    """Split the initial dataset into training (2/3) and prediction (1/3) sets"""
    try:
        # Create directories if they don't exist
        TRAINING_RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
        PREDICTION_RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
        
        # Read the initial dataset
        df = pd.read_csv(INITIAL_DATA_PATH)
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split index
        split_idx = int(len(df) * (2/3))
        
        # Split the data
        training_data = df[:split_idx]
        prediction_data = df[split_idx:]
        
        # Save the splits
        training_data.to_csv(TRAINING_RAW_DATA_PATH / 'weatherAUS.csv', index=False)
        prediction_data.to_csv(PREDICTION_RAW_DATA_PATH / 'weatherAUS.csv', index=False)
        
        return {
            'training_rows': len(training_data),
            'prediction_rows': len(prediction_data)
        }
    except Exception as e:
        print(f"Error in split_initial_dataset: {str(e)}")
        raise