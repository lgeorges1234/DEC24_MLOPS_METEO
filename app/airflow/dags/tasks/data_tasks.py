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
    Process the first row of the prediction dataset with additional debugging
    """
    import os
    import time
    import datetime
    
    print(f"Starting daily prediction row processing at {datetime.datetime.now()}")
    
    # Create directories if they don't exist
    TRAINING_RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    PREDICTION_RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    # Create a test file to verify we can write to the directory
    test_file = PREDICTION_RAW_DATA_PATH / 'test_write.txt'
    with open(test_file, 'w') as f:
        f.write(f"Test write at {datetime.datetime.now()}")
    
    print(f"Created test file at {test_file}, exists={test_file.exists()}")
    
    # Check file permissions on prediction directory
    prediction_dir = PREDICTION_RAW_DATA_PATH
    print(f"Prediction directory: {prediction_dir}")
    print(f"Directory exists: {prediction_dir.exists()}")
    print(f"Directory is readable: {os.access(prediction_dir, os.R_OK)}")
    print(f"Directory is writable: {os.access(prediction_dir, os.W_OK)}")
    
    # Read the prediction dataset
    prediction_file = PREDICTION_RAW_DATA_PATH / 'weatherAUS_prediction.csv'
    print(f"Prediction file path: {prediction_file}")
    print(f"File exists: {prediction_file.exists()}")
    if prediction_file.exists():
        print(f"File size: {prediction_file.stat().st_size} bytes")
        print(f"File permissions: {oct(prediction_file.stat().st_mode)}")
        print(f"File is readable: {os.access(prediction_file, os.R_OK)}")
        print(f"File is writable: {os.access(prediction_file, os.W_OK)}")
    
    # Make a backup of the prediction file for debugging
    if prediction_file.exists():
        backup_file = PREDICTION_RAW_DATA_PATH / f'weatherAUS_prediction_backup_{int(time.time())}.csv'
        with open(prediction_file, 'r') as src:
            content = src.read()
        with open(backup_file, 'w') as dst:
            dst.write(content)
        print(f"Created backup at {backup_file}")
    
    # Read the ENTIRE prediction dataset
    full_prediction_df = pd.read_csv(prediction_file)
    print(f"Read prediction file with {len(full_prediction_df)} rows")
    
    if full_prediction_df.empty:
        print("Warning: Prediction dataset is empty")
        return {"status": "empty_dataset"}
    
    # Print the first few rows to debug
    print("First 2 rows of prediction dataset:")
    print(full_prediction_df.head(2))
    
    # Extract the first row for processing
    prediction_df = full_prediction_df.iloc[[0]].copy()
    print(f"Extracted first row: {prediction_df.iloc[0]['Date']} for location {prediction_df.iloc[0]['Location']}")
    
    # 1. Append to training dataset
    training_file = TRAINING_RAW_DATA_PATH / 'weatherAUS_training.csv'
    prediction_df.to_csv(training_file, mode='a', header=not training_file.exists(), index=False)
    print(f"Added row to training dataset at {training_file}")
    
    # 2. Create a version for prediction
    daily_prediction_file = PREDICTION_RAW_DATA_PATH / 'daily_row_prediction.csv'
    prediction_df.to_csv(daily_prediction_file, index=False)
    print(f"Saved daily prediction row to {daily_prediction_file}")
    
    # 3. Remove the processed row from the prediction dataset and save
    updated_prediction_df = full_prediction_df.iloc[1:].copy()
    
    # Debug the updated dataframe
    print(f"Updated prediction dataset has {len(updated_prediction_df)} rows")
    if not updated_prediction_df.empty:
        print(f"New first row: {updated_prediction_df.iloc[0]['Date']} for location {updated_prediction_df.iloc[0]['Location']}")
    
    # Write to a new file first to avoid partial writes
    temp_file = PREDICTION_RAW_DATA_PATH / 'weatherAUS_prediction_new.csv'
    updated_prediction_df.to_csv(temp_file, index=False)
    print(f"Wrote updated data to temporary file {temp_file}")
    
    # Verify the temp file was written correctly
    if temp_file.exists():
        print(f"Temp file size: {temp_file.stat().st_size} bytes")
        # Read it back to double-check
        verification_df = pd.read_csv(temp_file)
        print(f"Verification: The temp file has {len(verification_df)} rows")
    
    # Now rename to replace the original file
    os.replace(temp_file, prediction_file)
    print(f"Replaced original file with updated version")
    
    # Verify the update worked
    if prediction_file.exists():
        verify_df = pd.read_csv(prediction_file)
        print(f"Final verification: The prediction file now has {len(verify_df)} rows")
        success = len(verify_df) == len(updated_prediction_df)
        print(f"Update successful: {success}")
    
    return {
        "daily_prediction_file": str(daily_prediction_file),
        "row_count": len(prediction_df),
        "remaining_prediction_rows": len(updated_prediction_df)
    }