#!/bin/bash

# Set environment name
ENV_NAME="hl2ss_env"

echo "Creating Conda environment: $ENV_NAME..."

# Instsll GTK dependencies
sudo apt-get update
sudo apt-get install libgtk2.0-dev pkg-config

# Create a Conda environment with Python 3.9
conda create -n $ENV_NAME python=3.9 -y

# Activate the environment
echo "Activating Conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install required packages
echo "Installing required packages..."
pip install opencv-python 
conda install -c conda-forge numpy opencv av pynput open3d pyaudio -y

# Print success message
echo "Environment setup is complete."
echo "To activate the environment, run: conda activate $ENV_NAME"

# Check installed packages
echo "Installed packages:"
conda list
