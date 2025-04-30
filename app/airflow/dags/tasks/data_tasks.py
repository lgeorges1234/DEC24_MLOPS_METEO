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
    3. Remove the processed row from the prediction dataset
    
    This function assumes:
    - The prediction dataset is at PREDICTION_RAW_DATA_PATH/weatherAUS_prediction.csv
    - The training dataset is at TRAINING_RAW_DATA_PATH/weatherAUS_training.csv
    - The daily prediction file will be saved at PREDICTION_RAW_DATA_PATH/daily_row_prediction.csv
    """
    from config import TRAINING_RAW_DATA_PATH, PREDICTION_RAW_DATA_PATH
    
    logger.info("Starting daily prediction row processing")
    
    # Create directories if they don't exist
    TRAINING_RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    PREDICTION_RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    # Read the prediction dataset
    prediction_file = PREDICTION_RAW_DATA_PATH / 'weatherAUS_prediction.csv'
    if not prediction_file.exists():
        raise FileNotFoundError(f"Prediction file not found at {prediction_file}")
    
    # Read the entire prediction dataset
    prediction_df = pd.read_csv(prediction_file)
    if prediction_df.empty:
        raise ValueError("Prediction dataset is empty")
    
    # Store the original row count for verification
    original_row_count = len(prediction_df)
    logger.info(f"Original prediction dataset has {original_row_count} rows")
    
    # Extract first row for processing
    first_row = prediction_df.iloc[0:1].copy()
    logger.info(f"Extracted first row from prediction dataset: {first_row.shape}")
    
    # Log the details of the row being processed
    if 'Date' in first_row.columns:
        date_str = first_row['Date'].values[0]
        logger.info(f"PROCESSING ROW DATE: {date_str}")
    
    # Log more details of the row for verification
    row_details = {}
    for column in first_row.columns:
        row_details[column] = str(first_row[column].values[0])
    
    logger.info(f"DAILY ROW DETAILS: {row_details}")
    
    # Print a clear separator for visibility in logs
    logger.info("="*50)
    
    # 1. Append to training dataset
    training_file = TRAINING_RAW_DATA_PATH / 'weatherAUS_training.csv'
    
    # Check if training file exists, if not, create it with header
    if not training_file.exists():
        logger.info(f"Training file not found at {training_file}, creating new file")
        first_row.to_csv(training_file, index=False)
    else:
        # Append to existing file without header
        first_row.to_csv(training_file, mode='a', header=False, index=False)
    
    logger.info(f"Added row to training dataset at {training_file}")
    
    # 2. Create a version for prediction
    daily_prediction_file = PREDICTION_RAW_DATA_PATH / 'daily_row_prediction.csv'
    first_row.to_csv(daily_prediction_file, index=False)
    
    logger.info(f"Saved daily prediction row to {daily_prediction_file}")
    
    # 3. Remove the processed row from the prediction dataset
    remaining_rows = prediction_df.iloc[1:].reset_index(drop=True)
    expected_new_count = original_row_count - 1
    
    # Check that we have the expected number of rows
    if len(remaining_rows) != expected_new_count:
        logger.error(f"Row removal error: Expected {expected_new_count} rows but got {len(remaining_rows)}")
        raise ValueError("Failed to properly remove the first row from the prediction dataset")
    
    # Save the updated prediction dataset
    try:
        # Check if we have write permissions
        if os.access(prediction_file, os.W_OK):
            remaining_rows.to_csv(prediction_file, index=False)
            logger.info(f"Updated prediction dataset saved with {len(remaining_rows)} rows")
        else:
            logger.error(f"No write permission for file: {prediction_file}")
            raise PermissionError(f"Cannot write to file: {prediction_file}")
            
        # Verify the file was updated correctly
        try:
            # Read the file again to check its contents
            verification_df = pd.read_csv(prediction_file)
            
            # Check row count
            if len(verification_df) != expected_new_count:
                logger.error(f"Verification failed: File has {len(verification_df)} rows, expected {expected_new_count}")
                raise ValueError("File verification failed: Row count mismatch")
                
            # Compare the first row date with what we expect
            if 'Date' in verification_df.columns and 'Date' in prediction_df.columns:
                expected_next_date = prediction_df.iloc[1]['Date'] if original_row_count > 1 else None
                actual_next_date = verification_df.iloc[0]['Date'] if not verification_df.empty else None
                
                if expected_next_date and actual_next_date and expected_next_date != actual_next_date:
                    logger.error(f"Verification failed: Expected next date {expected_next_date}, got {actual_next_date}")
                    raise ValueError("File verification failed: Date mismatch")
                    
            logger.info("File verification successful: Prediction dataset correctly updated")
            
        except Exception as ve:
            logger.error(f"Verification error: {str(ve)}")
            raise
    
    except Exception as e:
        logger.error(f"Error updating prediction file: {str(e)}")
        raise
    
    logger.info(f"Successfully processed daily prediction row. Remaining rows in dataset: {len(remaining_rows)}")
    
    # Format and return all the relevant information
    return {
        "daily_prediction_file": str(daily_prediction_file),
        "date": str(first_row['Date'].values[0]) if 'Date' in first_row.columns else "Unknown",
        "location": str(first_row['Location'].values[0]) if 'Location' in first_row.columns else "Unknown",
        "row_details": row_details,  # Include full row details in the return value
        "rows_before": original_row_count,
        "rows_after": len(remaining_rows),
        "row_removed": True
    }