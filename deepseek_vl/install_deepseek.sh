#!/bin/bash

# Create a new directory called dsap-cloud
mkdir -p dsap-cloud && cd dsap-cloud

# Create a virtual environment called dsap
python3 -m venv dsap

# Activate the virtual environment
source dsap/bin/activate

# Pull from the DeepSeek repository
git clone https://github.com/TopazLabs/DeepSeek-VL
cd DeepSeek-VL

# Install the contents using pip
pip install -e .

# Log into Hugging Face using HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
  echo "HUGGINGFACE_TOKEN environment variable not set. Exiting."
  exit 1
fi
echo "$HF_TOKEN" | huggingface-cli login --token

# Clone the DeepSeek-AutoPrompt repository to the dsap-cloud directory
cd ..
git clone https://huggingface.co/TopazLabs/DeepSeek-AutoPrompt

# Terminate successfully
echo "Setup completed successfully."
