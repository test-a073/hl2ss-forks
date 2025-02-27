#!/bin/bash

echo "Please make sure 'hl2ss_sample' is running and the Jetson device and Hololens are in the same network"
echo ""

# Run the first Python script
echo "Running 1_image_capture.py..."
python3 1_image_capture.py

echo "----------------------------------"

# Run the second Python script
echo "Running 2_voice_transcribe.py..."
python3 2_voice_transcribe.py

echo "----------------------------------"

# Run the third Python script
echo "Running 3_vlm_inference.py..."
python3 3_vlm_inference.py

echo "----------------------------------"
echo "All scripts have finished executing."
