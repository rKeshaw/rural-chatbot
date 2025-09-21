# Use Python 3.9 base image for compatibility
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install Whisper from GitHub
RUN pip install git+https://github.com/openai/whisper.git

# Install TTS (Coqui fork) from GitHub
RUN pip install git+https://github.com/coqui-ai/TTS.git

# Copy src code (empty for now, but keep structure)
COPY src/ /app/src/

# Set environment variable for Python buffering
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["bash"]

