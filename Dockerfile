# Build slim:
#  docker build --build-arg HF_TOKEN=your_huggingface_token -t deepseek-vl .
# 
# Build with Model:
#  docker build --build-arg HF_TOKEN=your_huggingface_token --build-arg PACKAGE_MODEL=1 -t deepseek-vl .
# 
# Run with:
#  docker run --gpus all -p 8560:8560 deepseek-vl
#
# Run with mounted model path:
#  docker run --gpus all -p 8560:8560 -v /path/to/model:/app/DeepSeek-AutoPrompt deepseek-vl

#
# docker build --build-arg HF_TOKEN=${HF_TKN} -t deepseek-vl:TAG .
#
# Deploy to AWS:
# aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 801793713466.dkr.ecr.us-east-1.amazonaws.com
# docker tag deepseek-vl:slim 801793713466.dkr.ecr.us-east-1.amazonaws.com/mleng/auto-prompt:slim
# docker push 801793713466.dkr.ecr.us-east-1.amazonaws.com/mleng/auto-prompt:slim

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
# Add build arg for package model
ARG PACKAGE_MODEL
ENV PACKAGE_MODEL=${PACKAGE_MODEL}

ARG MODEL_DIR=/data/autoprompt

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install DeepSeek-VL from GitHub
RUN pip3 install git+https://github.com/TopazLabs/DeepSeek-VL

# Install Hugging Face CLI
RUN pip3 install --upgrade huggingface_hub

# Add build arg for Hugging Face token
ARG HF_TOKEN
ENV HUGGINGFACE_TOKEN=${HF_TOKEN}

# Clone DeepSeek-AutoPrompt repository using HF CLI if PACKAGE_MODEL is set
RUN if [ ! -z "$PACKAGE_MODEL" ]; then \
    MODEL_DIR=/data/autoprompt && \
    huggingface-cli login --token ${HUGGINGFACE_TOKEN} && \
    huggingface-cli download TopazLabs/DeepSeek-AutoPrompt --local-dir ${MODEL_DIR} --token ${HUGGINGFACE_TOKEN}; \
    fi
    
RUN MODEL_DIR=/app/models/translation_model && \
    huggingface-cli login --token ${HUGGINGFACE_TOKEN} && \
    huggingface-cli download TopazLabs/Gemma-3-1b --local-dir ${MODEL_DIR} --token ${HUGGINGFACE_TOKEN}; 

# Copy application code
COPY . /app

# Expose port
EXPOSE 8560

# This will start the FastAPI server when you run: docker run -p 8560:8560 <image-name>
CMD ["deepseek-vl", "api", "--port", "8560", "--model-path", "/data/autoprompt"]
