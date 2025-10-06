# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BAAI Reranker Service - A containerized FastAPI web service for text reranking using PyTorch and Transformers with the `BAAI/bge-reranker-v2-m3` model. The model is pre-downloaded during Docker build for fast startup and offline operation.

## Build & Run Commands

```bash
cd reranker-service

# Build and run
make build                    # Build Docker image (downloads model during build)
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

### PyTorch + FastAPI Setup
- **PyTorch + Transformers**: Direct inference with `BAAI/bge-reranker-v2-m3` cross-encoder model
- **FastAPI Application**: REST API on port 8000
- **Pre-downloaded Model**: Baked into Docker image (~2.3GB) for fast startup

### Request Flow
```
Client → FastAPI (port 8000) → PyTorch Model → Response
```

### Startup Sequence
1. Container starts, `startup.sh` executes
2. Model loads from pre-downloaded cache (`/root/.cache/huggingface/`)
3. FastAPI application starts (15-30 seconds total)

## Key Files

```
reranker-service/
├── app/
│   └── main.py                  # FastAPI app with PyTorch integration
├── client/
│   └── test_client.py           # Test client
├── terraform/                   # Azure infrastructure as code
│   ├── main.tf                  # Infrastructure definition
│   ├── variables.tf             # Variable declarations
│   ├── outputs.tf               # Output definitions
│   └── terraform.tfvars.example # Example configuration
├── scripts/
│   └── deploy_azure.sh          # Azure deployment script
├── Dockerfile                   # Container build with model download
├── docker-compose.yml           # Production deployment config
├── startup.sh                   # Service startup script
├── requirements.txt             # Python dependencies (torch, transformers, fastapi)
├── Makefile                     # Build automation
└── README.md                    # Documentation
```

## API Endpoints

All endpoints documented at `http://localhost:8000/docs` (Swagger) or `/redoc` (ReDoc)

- **GET /health**: Health check for model status
  - Returns: `{status, model_loaded, model_name, device, version}`
- **POST /rerank**: Single query-passage pair reranking
  - Body: `{query, passage, normalize}`
  - Returns: `{score, normalized, query_length, passage_length}`
- **POST /rerank/batch**: Batch reranking (max 100 pairs)
  - Body: `{pairs: [{query, passage}], normalize}`
  - Returns: `{scores, normalized, pairs_count}`
  - Optimizes by grouping pairs with same query

## Environment Variables

```bash
MODEL_NAME=BAAI/bge-reranker-v2-m3  # HuggingFace model identifier
PORT=8000                            # FastAPI service port
HOST=0.0.0.0                         # Bind address
USE_FP16=false                       # Enable FP16 precision (requires CUDA)
CORS_ORIGINS=*                       # CORS allowed origins
```

## Development Workflow

### Local Development (without Docker)
```bash
cd reranker-service
pip install -r requirements.txt
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing Flow
1. Start service: `make run`
2. Wait 15-30 seconds for model loading
3. Test endpoints: `make test-local`
4. Test with client: `python client/test_client.py`
5. View logs: `make logs`

### Docker Build Details
- **Base Image**: `python:3.11-slim`
- **Model Download**: During build (RUN step)
- **Model Cache**: `/root/.cache/huggingface/` (baked into image + volume)
- **Python Environment**: Virtual environment at `/app/venv`
- **Startup**: Simple `startup.sh` launches FastAPI directly

## Important Implementation Details

### Rerank Model Class (`main.py`)
- **Cross-Encoder Architecture**: Processes `[query, passage]` pairs jointly
- **Score Extraction**: Gets logits from model output: `model(**inputs).logits.view(-1,).float()`
- **Device Selection**: Auto-detects CUDA vs CPU
- **FP16 Support**: Optional half-precision for GPU acceleration
- **Batch Processing**: Handles multiple passages per query efficiently
- **Tokenization**: Max length 512 tokens with truncation

### Model Loading
- Pre-downloaded during Docker build (no runtime download)
- Uses `AutoModelForSequenceClassification.from_pretrained()`
- Cached in `/root/.cache/huggingface/hub/`
- Persisted via Docker volume for faster rebuilds

### Request Validation
- Pydantic models validate all inputs
- Text length limits: 1-10,000 characters
- Batch size limit: 1-100 pairs
- Empty/whitespace validation

### Score Normalization
- Optional sigmoid normalization: `1 / (1 + exp(-score))`
- Converts raw logits to 0-1 range
- Applied consistently across single and batch endpoints

### Batch Optimization
- Groups requests by query to minimize model calls
- Processes all passages for same query in single forward pass
- Maps scores back to original positions in response

## Resource Requirements

- **Memory**: 2-4GB (model ~1.5GB + inference overhead)
- **CPU**: 1-2 cores
- **Disk**: 3GB for Docker image (includes pre-downloaded model)
- **Startup Time**: 15-30 seconds (model loading from cache)
- **GPU**: Optional (set `USE_FP16=true` for FP16 acceleration)

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
- Note: Model is baked into image, no external storage needed

## Troubleshooting

### Build Issues
- **Slow build**: Model download during build (~2.3GB)
  - Subsequent builds use Docker layer cache
  - Check: `docker build --progress=plain .`
- **Out of disk space**: Ensure 5GB+ free for build
- **Network errors**: Check DNS, retry build

### Startup Issues
- **Health check fails**: Model still loading (wait 30-60s)
- **Check logs**: `make logs` or `docker-compose logs -f`
- **Memory errors**: Increase container limits in `docker-compose.yml`

### Runtime Errors
- **503 Service Unavailable**: Model not loaded
  - Check: `curl localhost:8000/health`
  - View logs for PyTorch errors
- **Slow inference**: First request may be slower (JIT compilation)
- **OOM errors**: Reduce batch size or increase memory

### Model Management
```bash
# Check model status
curl http://localhost:8000/health

# View model loading logs
make logs

# Rebuild with updated model
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Health Check Debugging
```bash
# Check FastAPI
curl http://localhost:8000/health

# Check model info
curl http://localhost:8000/ | jq .

# Check container logs
make logs

# Interactive troubleshooting
docker exec -it baai-reranker-service /bin/bash
```

## Performance Considerations

1. **Pre-downloaded Model**: No runtime download = fast startup
2. **Batch Processing**: Group requests with same query for better throughput
3. **FP16 Precision**: Enable on GPU for 2x faster inference
4. **Resource Limits**: Configure appropriate CPU/memory in docker-compose.yml
5. **Model Caching**: Docker volume persists cache across container rebuilds

## Technical Details

### Cross-Encoder vs Bi-Encoder
- **This model**: Cross-encoder (processes query+passage jointly)
- **Output**: Single relevance score per pair
- **Not suitable for**: Large-scale retrieval (use bi-encoders/embeddings)
- **Best for**: Reranking top-K results from initial retrieval

### Model Architecture
- **Base**: XLM-RoBERTa (multilingual BERT)
- **Type**: Sequence classification (binary classification head)
- **Parameters**: 568M
- **Max Input**: 512 tokens (truncated if longer)
- **Output**: Single logit value (raw score)

### Score Interpretation
- **Raw scores**: Typically in range [-10, +10]
- **Higher = more relevant**
- **Normalized scores**: Sigmoid transform to [0, 1]
- **No fixed threshold**: Scores are relative, use for ranking

## Code Quality

Dependencies:
- `torch>=2.0.0` - PyTorch for model inference
- `transformers>=4.35.0` - HuggingFace Transformers
- `fastapi==0.104.1` - Web framework
- `uvicorn[standard]==0.24.0` - ASGI server
- `pydantic==2.5.0` - Data validation

No Ollama dependency - direct PyTorch implementation for accurate reranking.

## Migration from Ollama

This service was migrated from Ollama-based implementation because:
- Ollama v0.12.3 lacks native `/api/rerank` endpoint
- Cross-encoder models require direct logits access
- Ollama's `/api/generate` returned incorrect scores (constant 0.525)
- PyTorch implementation provides accurate, reliable reranking

### Key Changes
- Removed: Ollama server, httpx Ollama client
- Added: PyTorch, Transformers, direct model inference
- Model: Pre-downloaded during build (was runtime download)
- Scores: Now correct (was hardcoded fallback)
