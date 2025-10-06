#!/bin/bash
set -e

echo "🚀 Starting BAAI Reranker Service"

# Display configuration
echo "📋 Configuration:"
echo "   Model: ${MODEL_NAME}"
echo "   Port: ${PORT}"
echo "   Host: ${HOST}"
echo "   FP16: ${USE_FP16}"

# Check if model is cached
if [ -d "/root/.cache/huggingface/hub" ]; then
    echo "✅ Model cache found"
else
    echo "⚠️  Model cache not found - model will be downloaded on first request"
fi

# Start FastAPI application
echo "🌐 Starting FastAPI application..."
cd /app
exec /app/venv/bin/python -m uvicorn main:app --host ${HOST} --port ${PORT}
