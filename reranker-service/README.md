# BAAI Reranker Service

A containerized web service for text reranking using the `BAAI/bge-reranker-v2-m3` model with PyTorch and Transformers.

## 🚀 Quick Start

### Build and Run
```bash
# Build the container (downloads model during build)
make build

# Run with docker-compose
make run

# Or manually
docker-compose up -d
```

### Test the Service
```bash
# Wait for startup (model loading)
python client/test_client.py

# Or manual test
curl http://localhost:8000/health
```

## 🏗️ Architecture

### Service Architecture
- **FastAPI**: REST API interface on port 8000
- **PyTorch + Transformers**: Direct model inference
- **Pre-downloaded Model**: Baked into Docker image (~2.3GB)

### Service Flow
```
Client → FastAPI (port 8000) → PyTorch Model → Response
```

## 📁 Key Files

### Core Files
- `Dockerfile` - Container setup with pre-downloaded model
- `docker-compose.yml` - Deployment configuration
- `startup.sh` - Service startup script
- `requirements.txt` - Python dependencies
- `app/main.py` - FastAPI application with PyTorch integration
- `client/test_client.py` - Test client

## 🔧 Configuration

### Environment Variables
```bash
MODEL_NAME=BAAI/bge-reranker-v2-m3
PORT=8000
HOST=0.0.0.0
USE_FP16=false  # Enable FP16 precision (requires CUDA)
CORS_ORIGINS=*
```

### Resource Requirements
- **Memory**: 2-4GB (model + inference)
- **CPU**: 1-2 cores
- **Disk**: 3GB for model storage (baked into image)
- **Startup Time**: 15-30 seconds (model loading from cache)

## 🚦 Startup Process

1. **Container Start**: Startup script executes
2. **Model Loading**: Loads pre-downloaded model from `/root/.cache/huggingface/`
3. **FastAPI Start**: REST API becomes available
4. **Health Check**: Service monitored

## 📊 API Endpoints

- `GET /health` - Health check (model status)
- `POST /rerank` - Single text pair
- `POST /rerank/batch` - Multiple pairs
- `GET /docs` - API documentation

### Example Usage
```bash
# Single rerank
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "passage": "AI learns from data",
    "normalize": true
  }'

# Batch rerank
curl -X POST http://localhost:8000/rerank/batch \
  -H "Content-Type: application/json" \
  -d '{
    "pairs": [
      {"query": "python", "passage": "programming language"},
      {"query": "python", "passage": "snake species"}
    ],
    "normalize": true
  }'
```

## 🐛 Troubleshooting

### Common Issues

**Container takes long to build**
- Model download happens during build (~2.3GB)
- Subsequent builds use Docker cache
- Check logs: `docker build --progress=plain .`

**Health check fails**
- Model may still be loading into memory
- Wait 30-60 seconds for full startup
- Check logs: `docker-compose logs -f` or `make logs`

**Out of memory**
- Increase container memory limit in docker-compose.yml
- Model requires ~2GB RAM minimum
- Disable FP16 if using CPU: `USE_FP16=false`

### Monitoring
```bash
# View logs
docker-compose logs -f
# Or use makefile
make logs

# Check FastAPI
curl http://localhost:8000/health
```

## 🚀 Deployment

### Azure Deployment
Deploy using Terraform infrastructure as code:
```bash
# Update terraform variables
cd terraform
# Edit terraform.tfvars with your Azure configuration
terraform apply
```

### Performance Benefits
- ✅ **No runtime downloads**: Model pre-downloaded in image
- ✅ **Fast inference**: PyTorch optimizations
- ✅ **Accurate scores**: Proper cross-encoder implementation
- ✅ **Better resource usage**: Direct model access

## 🔄 Model Management

### Model Cache
The model is pre-downloaded during Docker build and cached in:
- **Build time**: `/root/.cache/huggingface/` (baked into image)
- **Runtime**: Persisted via Docker volume `huggingface_cache`

### Model Updates
```bash
# Rebuild with latest model version
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 📝 Technical Details

### Model Information
- **Name**: BAAI/bge-reranker-v2-m3
- **Type**: Cross-encoder reranker
- **Parameters**: 568M
- **Input**: Query-passage pairs
- **Output**: Relevance scores (logits)
- **Max Length**: 512 tokens

### How It Works
Unlike embedding models that produce vectors, this cross-encoder:
1. Takes `[query, passage]` as input
2. Processes them jointly through BERT-based model
3. Outputs a single relevance score from the classification head
4. Optional sigmoid normalization to [0,1] range

### No HuggingFace API Key Required
The model is public and requires no authentication for download or use.
