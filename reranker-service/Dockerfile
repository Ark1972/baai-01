# BAAI Reranker Service Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_NAME=BAAI/bge-reranker-v2-m3 \
    PORT=8000 \
    HOST=0.0.0.0 \
    USE_FP16=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app /root/.cache/huggingface

# Set working directory
WORKDIR /app

# Create virtual environment and install Python dependencies
RUN python -m venv /app/venv && \
    /app/venv/bin/pip install --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt /app/
# Install PyTorch CPU-only version first (smaller and faster than default CUDA version)
RUN /app/venv/bin/pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu
# Install remaining dependencies
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Pre-download the model during build to avoid runtime download
# This caches the model in /root/.cache/huggingface/
RUN /app/venv/bin/python -c "\
from FlagEmbedding import FlagReranker; \
print('Downloading BAAI reranker model...'); \
FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False); \
print('Model downloaded successfully')"

# Copy application code and startup script
COPY ./app /app
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run startup script
ENTRYPOINT ["/bin/bash"]
CMD ["/app/startup.sh"]
