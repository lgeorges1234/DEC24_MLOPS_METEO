#!/bin/bash

# Create the main directories structure
mkdir -p raw_data/initial_dataset
mkdir -p raw_data/training_raw_data
mkdir -p raw_data/prediction_raw_data

# Set permissions for Airflow and API to both access these directories
# Using the root group for permissions
AIRFLOW_GID=0

# Set ownership and permissions
chown -R ubuntu:${AIRFLOW_GID} raw_data
chmod -R 775 raw_data

# Make new files inherit group ownership
find raw_data -type d -exec chmod g+s {} \;

echo "Directory structure created successfully"
echo "You can now place your initial dataset in raw_data/initial_dataset/"