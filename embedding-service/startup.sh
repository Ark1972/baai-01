#!/bin/bash

echo "Starting BAAI BGE-M3 Embedding Service..."
echo "Model: ${MODEL_NAME:-BAAI/bge-m3}"
echo "Port: ${PORT:-8001}"
echo "FP16: ${USE_FP16:-false}"

# Activate virtual environment
source /app/venv/bin/activate

# Start FastAPI application
exec uvicorn main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8001}
