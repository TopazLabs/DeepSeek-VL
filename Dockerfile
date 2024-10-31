FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

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

# Change into the deepseek repo directory
WORKDIR /app/DeepSeek-VL

# Copy and install DeepSeek-VL from local directory
COPY . /app/DeepSeek-VL
RUN pip3 install /app/DeepSeek-VL

# Install Python dependencies
COPY requirements.txt /app/DeepSeek-VL
RUN pip3 install --no-cache-dir -r /app/DeepSeek-VL/requirements.txt

# Copy application code
COPY . /app/DeepSeek-VL

# Expose port
EXPOSE 8560

# Run the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"] 