# BAAI Reranker Service Dockerfile
FROM ollama/ollama:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    MODEL_NAME=xitao/bge-reranker-v2-m3 \
    PORT=8000 \
    HOST=0.0.0.0

# Install Python and supervisor for service management
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Create necessary directories
RUN mkdir -p /app /var/log/supervisor /root/.ollama

# Set working directory
WORKDIR /app

# Create virtual environment and install Python dependencies
RUN python -m venv /app/venv && \
    /app/venv/bin/pip install --upgrade pip

# Copy minimal requirements and install
COPY requirements.txt /app/
RUN /app/venv/bin/pip install -r requirements.txt

# Copy application code and startup script
COPY ./app /app
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Note: Model will be pulled on first startup to avoid build complexity

# Expose ports
EXPOSE 8000 11434

# Health check for both services
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health && curl -f http://localhost:11434/api/tags || exit 1

# Override Ollama's default entrypoint and use our startup script
ENTRYPOINT ["/bin/bash"]
CMD ["/app/startup.sh"]