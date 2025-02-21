import pandas as pd
from pathlib import Path
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
        # df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.sort_values('Date')
        
        # Calculate split index
        split_idx = int(len(df) * (2/3))
        
        # Split the data
        training_data = df[:split_idx]
        prediction_data = df[split_idx:]
        
        # Save the splits
        training_data.to_csv(TRAINING_RAW_DATA_PATH / 'weatherAUS_training.csv', index=False)
        prediction_data.to_csv(PREDICTION_RAW_DATA_PATH / 'weatherAUS_prediction.csv', index=False)
        
        return {
            'training_rows': len(training_data),
            'prediction_rows': len(prediction_data)
        }
    except Exception as e:
        print(f"Error in split_initial_dataset: {str(e)}")
        raise

def process_daily_prediction_row(**context):
    """
    Process the first row of the prediction dataset:
    1. Add it to the training dataset
    2. Create a version without the target (RainTomorrow) for prediction
    
    This function assumes:
    - The prediction dataset is at PREDICTION_RAW_DATA_PATH/weatherAUS_prediction.csv
    - The training dataset is at TRAINING_RAW_DATA_PATH/weatherAUS_training.csv
    - The daily prediction file will be saved at PREDICTION_RAW_DATA_PATH/daily_row_prediction.csv
    """
    print(f"Starting daily prediction row processing")
    
    # Create directories if they don't exist
    TRAINING_RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    PREDICTION_RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    # Read the prediction dataset
    prediction_file = PREDICTION_RAW_DATA_PATH / 'weatherAUS_prediction.csv'
    if not prediction_file.exists():
        raise FileNotFoundError(f"Prediction file not found at {prediction_file}")
    
    # Read only the first row
    prediction_df = pd.read_csv(prediction_file, nrows=1)
    if prediction_df.empty:
        raise ValueError("Prediction dataset is empty")
    
    print(f"Extracted first row from prediction dataset: {prediction_df.shape}")
    
    # 1. Append to training dataset
    training_file = TRAINING_RAW_DATA_PATH / 'weatherAUS_training.csv'
    
    # Check if training file exists, if not, create it with header
    if not training_file.exists():
        print(f"Training file not found at {training_file}, creating new file")
        prediction_df.to_csv(training_file, index=False)
    else:
        # Append to existing file without header
        prediction_df.to_csv(training_file, mode='a', header=False, index=False)
    
    print(f"Added row to training dataset at {training_file}")
    
    # 2. Create a version without the target for prediction
    # Assuming 'RainTomorrow' is the target column
    if 'RainTomorrow' in prediction_df.columns:
        prediction_df_no_target = prediction_df.drop(columns=['RainTomorrow'])
    else:
        print("Warning: 'RainTomorrow' column not found in dataset")
        prediction_df_no_target = prediction_df.copy()
    
    # Save the prediction row without target
    daily_prediction_file = PREDICTION_RAW_DATA_PATH / 'daily_row_prediction.csv'
    prediction_df_no_target.to_csv(daily_prediction_file, index=False)
    
    print(f"Saved daily prediction row without target to {daily_prediction_file}")
    
    return {
        "daily_prediction_file": str(daily_prediction_file),
        "row_count": len(prediction_df)
    }    