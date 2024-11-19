#!/bin/bash

# Create a new directory called dsap-cloud
mkdir -p dsap-cloud && cd dsap-cloud

# Create a virtual environment called dsap
python3 -m venv dsap

# Activate the virtual environment
source dsap/bin/activate

# Install from the DeepSeek repository
pip install git+https://github.com/TopazLabs/DeepSeek-VL

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

# cd dsap-cloud && source dsap/bin/activate && deepseek-vl api --model-path ./DeepSeek-AutoPrompt/