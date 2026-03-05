# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install RAHL in development mode
RUN pip install -e .

# Create cache directory for models
RUN mkdir -p /root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface

# Test import (this will work now because RAHL is installed)
RUN python -c "from rahl import RAHLPipeline; print('✓ RAHL successfully imported')"

# Expose port
EXPOSE 10000

# Start command
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "2", "--timeout", "300"]
