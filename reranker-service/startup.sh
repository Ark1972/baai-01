#!/bin/bash
set -e

echo "🚀 Starting BAAI Reranker Service"

# Function to download model with retries and resume
download_model_with_retries() {
    local model_name="$1"
    local max_retries=5
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        echo "📥 Attempt $((retry_count + 1))/$max_retries: Downloading $model_name..."
        
        # Set environment variables for better download performance
        export OLLAMA_MAX_LOADED_MODELS=1
        export OLLAMA_NUM_PARALLEL=1
        export OLLAMA_FLASH_ATTENTION=false
        
        # Start download with timeout
        if timeout 1800 ollama pull "$model_name"; then  # 30 minute timeout
            echo "✅ Model downloaded successfully"
            return 0
        else
            retry_count=$((retry_count + 1))
            echo "⚠️  Download failed, retry $retry_count/$max_retries in 10 seconds..."
            sleep 10
            
            # Clean up any partial downloads
            ollama rm "$model_name" 2>/dev/null || true
        fi
    done
    
    echo "❌ Failed to download model after $max_retries attempts"
    return 1
}

# Start Ollama server in background
echo "📡 Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama server is ready"
        break
    fi
    echo "   Attempt $i/30 - waiting..."
    sleep 2
done

# Check if server is ready
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama server failed to start"
    exit 1
fi

# Check if model is already available (from mounted volume or previous download)
echo "🔍 Checking for existing model: ${MODEL_NAME}"
if ollama list | grep -q "${MODEL_NAME}"; then
    echo "✅ Model already available, skipping download"
elif [ -d "/app/models/${MODEL_NAME}" ]; then
    echo "✅ Model found in mounted directory"
    # Import model from mounted directory if needed
    # (This would require additional logic to convert from HF format to Ollama format)
else
    echo "📥 Model not found, downloading..."
    
    # Try optimized download with retries
    if ! download_model_with_retries "${MODEL_NAME}"; then
        echo "❌ Failed to download model, trying alternative..."
        
        # Try smaller alternative model as fallback
        FALLBACK_MODEL="nomic-embed-text"
        echo "🔄 Trying fallback model: $FALLBACK_MODEL"
        if download_model_with_retries "$FALLBACK_MODEL"; then
            echo "⚠️  Using fallback model: $FALLBACK_MODEL"
            export MODEL_NAME="$FALLBACK_MODEL"
        else
            echo "❌ All download attempts failed"
            exit 1
        fi
    fi
fi

# Verify model is available
echo "🔍 Verifying model availability..."
if ! ollama list | grep -q "${MODEL_NAME}"; then
    echo "❌ Model verification failed"
    exit 1
fi

echo "✅ Model verification successful"

# Start FastAPI
echo "🌐 Starting FastAPI application..."
cd /app
exec /app/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000