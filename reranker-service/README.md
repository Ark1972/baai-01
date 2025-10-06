# BAAI Reranker Service

A containerized web service for text reranking using the `xitao/bge-reranker-v2-m3` model. This service combines Ollama's optimized inference engine with a FastAPI REST interface.

## 🚀 Quick Start

### Build and Run
```bash
# Build the container
make build

# Run with docker-compose
make run

# Or manually
docker-compose up -d
```

### Test the Service
```bash
# Wait for startup (model download + loading)
python client/test_client.py

# Or manual test
curl http://localhost:8000/health
```

## 🏗️ Architecture

### Service Architecture
- **Ollama Server**: Runs `xitao/bge-reranker-v2-m3` model on port 11434
- **FastAPI**: Provides REST API interface on port 8000
- **Startup Script**: Manages service orchestration in single container

### Service Flow
```
Client → FastAPI (port 8000) → Ollama API (port 11434) → Model → Response
```

## 📁 Key Files

### Core Files
- `Dockerfile` - Container setup
- `docker-compose.yml` - Deployment configuration
- `startup.sh` - Service orchestration script
- `requirements.txt` - Python dependencies
- `app/main.py` - FastAPI application
- `client/test_client.py` - Test client

## 🔧 Configuration

### Environment Variables
```bash
MODEL_NAME=xitao/bge-reranker-v2-m3
OLLAMA_HOST=127.0.0.1:11434
PORT=8000
HOST=0.0.0.0
CORS_ORIGINS=*
```

### Resource Requirements
- **Memory**: 2-4GB (model + inference)
- **CPU**: 1-2 cores
- **Disk**: 2GB for model storage
- **Startup Time**: 2-3 minutes (model download + load)

## 🚦 Startup Process

1. **Container Start**: Startup script executes
2. **Ollama Server**: Launches in background
3. **Model Download**: `ollama pull xitao/bge-reranker-v2-m3` with retry logic
4. **FastAPI**: Starts after Ollama and model are ready
5. **Health Check**: Both services monitored

## 📊 API Endpoints

- `GET /health` - Health check (both services)
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

**Container takes long to start**
- Model download happens during first run
- Check logs: `docker-compose logs -f` or `make logs`

**Health check fails**
- Ollama server may still be loading model
- Wait 2-3 minutes for full startup
- Check both services: `curl localhost:11434/api/tags`

**Out of memory**
- Increase container memory limit in docker-compose.yml
- Model requires ~1.5GB RAM minimum

### Monitoring
```bash
# View logs
docker-compose logs -f
# Or use makefile
make logs

# Check Ollama directly
curl http://localhost:11434/api/tags

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
- ✅ **No large downloads**: Ollama manages models efficiently
- ✅ **Optimized inference**: Ollama's runtime optimizations
- ✅ **Fast startup**: No PyTorch compilation
- ✅ **Better resource usage**: Shared model memory

## 🔄 Model Management

### Available Commands
```bash
# Inside container
docker exec -it baai-reranker-service ollama list
docker exec -it baai-reranker-service ollama pull xitao/bge-reranker-v2-m3
docker exec -it baai-reranker-service ollama rm xitao/bge-reranker-v2-m3
```

### Model Updates
```bash
# Pull latest model version
docker exec -it baai-reranker-service ollama pull xitao/bge-reranker-v2-m3

# Restart container to use updated model
docker-compose restart
```