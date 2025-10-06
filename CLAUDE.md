# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BAAI Reranker Service - A containerized FastAPI web service for text reranking using Ollama with the `xitao/bge-reranker-v2-m3` model. This hybrid architecture combines Ollama's optimized inference engine with a FastAPI REST interface for production-ready text reranking.

## Build & Run Commands

```bash
cd reranker-service

# Build and run
make build                    # Build Ollama Docker image
make run                      # Start with docker-compose
make stop                     # Stop services
make logs                     # View logs

# Testing
make test-local               # Test API endpoints with curl
make test-client              # Run Python client test
make install-deps             # Install Python dependencies

# Development
make dev                      # Run with hot reload (local)

# Azure deployment
make azure-login              # Login to Azure
make azure-acr-build          # Build in Azure Container Registry
make azure-acr-push           # Push to ACR
make tf-init                  # Initialize Terraform
make tf-plan                  # Preview infrastructure changes
make tf-apply                 # Deploy infrastructure
make deploy                   # Full pipeline: build + push + deploy
```

## Architecture

### Hybrid Ollama + FastAPI Setup
- **Ollama Server**: Manages and serves the `xitao/bge-reranker-v2-m3` model on port 11434
- **FastAPI Application**: Provides REST API on port 8000, proxies requests to Ollama
- **Startup Script**: `startup.sh` manages service orchestration with retry logic for model downloads

### Request Flow
```
Client → FastAPI (port 8000) → Ollama API (port 11434) → Model inference → Response
```

### Startup Sequence
1. Container starts, `startup.sh` executes
2. Ollama server launches in background
3. Script waits for Ollama readiness (up to 60s)
4. Model download with retry logic (5 attempts, 30min timeout each)
5. Model verification
6. FastAPI application starts

## Key Files

```
reranker-service/
├── app/
│   └── main.py                  # FastAPI app with Ollama integration
├── client/
│   └── test_client.py           # Test client
├── terraform/                   # Azure infrastructure as code
│   ├── main.tf                  # Infrastructure definition
│   ├── variables.tf             # Variable declarations
│   ├── outputs.tf               # Output definitions
│   └── terraform.tfvars.example # Example configuration
├── scripts/
│   └── deploy_azure.sh          # Azure deployment script
├── Dockerfile                   # Container build
├── docker-compose.yml           # Production deployment config
├── startup.sh                   # Service orchestration with retry logic
├── requirements.txt             # Python dependencies
├── Makefile                     # Build automation
└── README.md                    # Documentation
```

## API Endpoints

All endpoints documented at `http://localhost:8000/docs` (Swagger) or `/redoc` (ReDoc)

- **GET /health**: Health check for both FastAPI and Ollama services
  - Returns: `{status, ollama_status, model_name, version}`
- **POST /rerank**: Single query-passage pair reranking
  - Body: `{query, passage, normalize}`
  - Returns: `{score, normalized, query_length, passage_length}`
- **POST /rerank/batch**: Batch reranking (max 100 pairs)
  - Body: `{pairs: [{query, passage}], normalize}`
  - Returns: `{scores, normalized, pairs_count}`
  - Optimizes by grouping pairs with same query

## Environment Variables

```bash
MODEL_NAME=xitao/bge-reranker-v2-m3   # Ollama model identifier
OLLAMA_HOST=127.0.0.1:11434            # Ollama server address
PORT=8000                              # FastAPI service port
HOST=0.0.0.0                           # Bind address
CORS_ORIGINS=*                         # CORS allowed origins

# Ollama performance optimizations (set in docker-compose.yml)
OLLAMA_MAX_LOADED_MODELS=1             # Limit concurrent models
OLLAMA_NUM_PARALLEL=1                  # Sequential processing
OLLAMA_FLASH_ATTENTION=false           # Disable flash attention
OLLAMA_KEEP_ALIVE=5m                   # Model unload timeout
```

## Development Workflow

### Local Development (without Docker)
```bash
# Install Ollama first: https://ollama.ai
ollama serve &
ollama pull xitao/bge-reranker-v2-m3

cd reranker-service
pip install -r requirements.ollama.txt
cd app
uvicorn main_ollama:app --reload --host 0.0.0.0 --port 8000
```

### Testing Flow
1. Start service: `make run`
2. Wait 2-3 minutes for model download (first run only)
3. Test endpoints: `make test-local`
4. Test with client: `python client/test_ollama_client.py`
5. View logs: `make logs`

### Docker Build Details
- **Base Image**: `ollama/ollama:latest`
- **Additional Installs**: Python 3, pip, supervisor, curl
- **Python Environment**: Virtual environment at `/app/venv`
- **Model Storage**: Volume mounted at `/root/.ollama` for persistence
- **Startup**: Custom `startup.sh` script handles model download and service orchestration

## Important Implementation Details

### Model Download with Retry Logic
The `startup.sh` script implements robust model downloading:
- 5 retry attempts with 10-second delays
- 30-minute timeout per attempt
- Automatic cleanup of partial downloads
- Fallback to `nomic-embed-text` model if primary fails
- Environment variable configuration for Ollama performance

### Ollama Client (`main.py`)
- **Async Operations**: Uses `httpx.AsyncClient` for non-blocking I/O
- **Health Checks**: Verifies both Ollama server and model availability
- **Flexible Model Matching**: Handles model names with/without version tags
- **Batch Optimization**: Groups requests by query to minimize Ollama API calls
- **Error Handling**: Comprehensive exception handling with logging
- **Lifespan Management**: Waits for Ollama readiness before accepting requests

### Request Validation
- Pydantic models validate all inputs
- Text length limits: 1-10,000 characters
- Batch size limit: 1-100 pairs
- Empty/whitespace validation

### Score Normalization
- Optional sigmoid normalization: `1 / (1 + exp(-score))`
- Converts raw scores to 0-1 range
- Applied consistently across single and batch endpoints

## Resource Requirements

- **Memory**: 2-4GB (1.5GB for model + 0.5-2GB for inference)
- **CPU**: 1-2 cores
- **Disk**: 2GB for model storage (persistent volume recommended)
- **Startup Time**: 2-3 minutes on first run (model download), <30s on subsequent runs

## Azure Deployment

### Terraform Workflow
1. Configure variables in `terraform/terraform.tfvars`
2. Initialize: `make tf-init`
3. Preview: `make tf-plan`
4. Deploy: `make tf-apply`
5. Get outputs: `make tf-output`

### Infrastructure Components
- Azure Container Registry (ACR) for image storage
- Azure Container Instance (ACI) for deployment
- Resource Group for organization
- Optional: Storage account for model caching

### Deployment Script
`scripts/deploy_azure.sh` provides automated deployment:
- Creates resource group
- Sets up container registry
- Builds and pushes image to ACR
- Deploys container instance
- Configures networking and DNS

## Troubleshooting

### Startup Issues
- **Slow first start**: Model download can take 2-3 minutes
- **Check logs**: `make logs` or `docker-compose logs -f`
- **Ollama not ready**: Wait full 60s for server initialization
- **Model download fails**: Check DNS configuration (8.8.8.8, 1.1.1.1 configured)

### Runtime Errors
- **503 Service Unavailable**: Ollama server or model not loaded
  - Check: `curl localhost:11434/api/tags`
  - Verify model: `docker exec <container> ollama list`
- **Memory issues**: Increase container memory limit in `docker-compose.yml`
- **Slow inference**: Model still loading, wait additional 30s

### Model Management
```bash
# List loaded models
docker exec baai-reranker-service ollama list

# Pull/update model
docker exec baai-reranker-service ollama pull xitao/bge-reranker-v2-m3

# Remove model
docker exec baai-reranker-service ollama rm xitao/bge-reranker-v2-m3
```

### Health Check Debugging
```bash
# Check FastAPI
curl http://localhost:8000/health

# Check Ollama directly
curl http://localhost:11434/api/tags

# Check container logs
make logs

# Interactive troubleshooting
docker exec -it baai-reranker-service /bin/bash
```

## Performance Considerations

1. **Model Caching**: Use Docker volumes to persist Ollama models across restarts
2. **Batch Processing**: Group requests with same query for better throughput
3. **Resource Limits**: Configure appropriate CPU/memory limits in docker-compose.yml
4. **Startup Optimization**: Pre-downloaded models in volume reduce startup time to <30s
5. **Network Optimization**: DNS configuration (8.8.8.8) improves model download reliability

## Code Quality

Dependencies are minimal and focused:
- `fastapi==0.104.1` - Web framework
- `uvicorn[standard]==0.24.0` - ASGI server
- `pydantic==2.5.0` - Data validation
- `httpx==0.25.1` - Async HTTP client for Ollama
- `python-multipart==0.0.6` - Form data parsing
- `python-dotenv==1.0.0` - Environment config

No heavy ML dependencies (torch, transformers) - Ollama handles all inference.
