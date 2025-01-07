#!/bin/bash

# Run the first Python script
echo "Running 001_sasika_image_saver.py..."
python 001_sasika_image_saver.py

echo "----------------------------------"

# Run the second Python script
echo "Running 002_main_pipeline.py..."
python 002_main_pipeline.py

echo "----------------------------------"

# Run the third Python script
echo "Running jetson_client_main.py..."
python jetson_client_main.py

echo "----------------------------------"
echo "All scripts have finished executing."
