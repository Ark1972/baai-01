# Multi-stage build for optimal image size
# Stage 1: Builder stage
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /build

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    git \
    curl \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Test network connectivity
RUN echo "Testing network connectivity..." && \
    curl -v --connect-timeout 10 https://pypi.org/ && \
    echo "PyPI connectivity: OK"

# Copy pip configuration for better network handling
COPY pip.conf /etc/pip.conf

# Copy requirements file
COPY requirements.txt .

# Upgrade pip first with network resilience
RUN pip install --upgrade pip --timeout 60 --retries 5

# Install core dependencies first (smaller, more reliable)
RUN pip install --user --timeout 60 --retries 5 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    python-multipart==0.0.6 \
    python-dotenv==1.0.0

# Install ML dependencies separately (these are the large ones)
RUN pip install --user --timeout 120 --retries 3 \
    torch>=2.0.0 \
    transformers>=4.35.0

# Install remaining dependencies
RUN pip install --user --timeout 60 --retries 5 \
    FlagEmbedding==1.2.5 \
    sentence-transformers>=2.2.0 \
    httpx==0.25.1

# Install test dependencies
RUN pip install --user --timeout 60 --retries 5 \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1

# Stage 2: Runtime stage
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_CACHE_DIR=/app/models \
    MODEL_NAME=BAAI/bge-reranker-v2-m3 \
    USE_FP16=true \
    PORT=8000 \
    HOST=0.0.0.0

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser ./app /app

# Update PATH to include user's local bin
ENV PATH=/home/appuser/.local/bin:$PATH

# Switch to non-root user
USER appuser

# Pre-download the model during build (optional, increases image size but faster startup)
# Uncomment the following lines if you want to include the model in the image
# RUN python -c "from FlagEmbedding import FlagReranker; \
#      FlagReranker('${MODEL_NAME}', use_fp16=True, cache_dir='${MODEL_CACHE_DIR}')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
