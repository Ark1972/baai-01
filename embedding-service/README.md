# BAAI BGE-M3 Embedding Service

A containerized FastAPI web service for generating text embeddings using the BAAI/bge-m3 model with FlagEmbedding. The model is pre-downloaded during Docker build for fast startup and offline operation.

## Features

- **Model**: BAAI/bge-m3 (1024-dimensional embeddings)
- **Languages**: 100+ languages (multilingual, cross-lingual)
- **Max Input**: 8192 tokens (~50000 characters)
- **Batch Processing**: Up to 100 texts per request
- **Pre-loaded Model**: Baked into Docker image (~2GB)
- **Fast Startup**: 15-30 seconds (no runtime download)
- **Offline Operation**: Works without internet after build

## Quick Start

```bash
cd embedding-service

# Build Docker image (downloads model during build)
make build

# Start service on port 8001
make run

# Test endpoints
make test-local

# View logs
make logs

# Stop service
make stop
```

## API Endpoints

All endpoints documented at `http://localhost:8001/docs` (Swagger) or `/redoc` (ReDoc)

### **GET /health**
Health check for model status

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "BAAI/bge-m3",
  "device": "cpu",
  "version": "1.0.0",
  "dimensions": 1024
}
```

### **POST /embed**
Generate embeddings for texts

**Request**:
```json
{
  "texts": [
    "First text to embed",
    "Second text to embed"
  ]
}
```

**Response**:
```json
{
  "embeddings": [
    [0.023, -0.145, 0.892, ...],  // 1024 floats
    [0.112, -0.034, 0.456, ...]   // 1024 floats
  ],
  "dimensions": 1024,
  "texts_count": 2,
  "model": "BAAI/bge-m3"
}
```

## Environment Variables

```bash
MODEL_NAME=BAAI/bge-m3          # HuggingFace model identifier
PORT=8001                        # FastAPI service port (8001 for embeddings, 8000 for reranker)
HOST=0.0.0.0                     # Bind address
USE_FP16=false                   # Enable FP16 precision (requires CUDA)
CORS_ORIGINS=*                   # CORS allowed origins
```

## Development Workflow

### Local Development (without Docker)
```bash
cd embedding-service
pip install -r requirements.txt
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### Testing Flow
1. Start service: `make run`
2. Wait 15-30 seconds for model loading
3. Test endpoints: `make test-local`
4. Test with client: `python client/test_client.py`
5. View logs: `make logs`

## Docker Build Details

- **Base Image**: `python:3.11-slim`
- **Model Download**: During build (RUN step)
- **Model Cache**: `/root/.cache/huggingface/` (baked into image + volume)
- **Python Environment**: Virtual environment at `/app/venv`
- **Startup**: Simple `startup.sh` launches FastAPI directly

## BGE-M3 Model Details

### Architecture
- **Base**: XLM-RoBERTa (multilingual BERT)
- **Parameters**: ~568M
- **Dimensions**: 1024
- **Max Tokens**: 8192
- **Languages**: 100+

### Multi-Functionality
BGE-M3 supports three retrieval methods:
1. **Dense Retrieval** (default in this service)
2. **Sparse Retrieval** (lexical matching)
3. **Multi-Vector Retrieval** (ColBERT-style)

This service implements dense retrieval. Full multi-functionality requires additional configuration.

### Performance
- **CPU**: ~0.5-2s per text (depends on length)
- **GPU (FP16)**: ~0.1-0.5s per text (5-10x faster)
- **Batch Processing**: More efficient than individual requests

## Resource Requirements

- **Memory**: 2-4GB (model ~2GB + inference overhead)
- **CPU**: 1-2 cores
- **Disk**: 3GB for Docker image (includes pre-downloaded model)
- **Startup Time**: 15-30 seconds (model loading from cache)
- **GPU**: Optional (set `USE_FP16=true` for FP16 acceleration)

## Use Cases

### 1. Semantic Search
```python
import requests

# Generate embeddings for documents
docs = ["Document 1", "Document 2", "Document 3"]
response = requests.post("http://localhost:8001/embed", json={"texts": docs})
doc_embeddings = response.json()["embeddings"]

# Store in vector database (Pinecone, Weaviate, etc.)
# Search with query embedding
```

### 2. Two-Stage Retrieval (with Reranker Service)
```python
# Stage 1: Fast embedding search
query_emb = requests.post("http://localhost:8001/embed",
                          json={"texts": [query]}).json()["embeddings"][0]
top_100 = vector_db.search(query_emb, k=100)

# Stage 2: Precise reranking
reranked = requests.post("http://localhost:8000/rerank/query",
                         json={"query": query, "passages": top_100})
top_20 = reranked.json()["re_ranked"][:20]
```

### 3. Multilingual Similarity
```python
# German and English texts are comparable in embedding space
texts = [
    "Die Steuervorteile für Selbständige",
    "Tax advantages for self-employed individuals"
]
embeddings = requests.post("http://localhost:8001/embed",
                          json={"texts": texts}).json()["embeddings"]

# Compute cosine similarity for cross-lingual matching
```

## Troubleshooting

### Build Issues
- **Slow build**: Model download during build (~2GB)
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
  - Check: `curl localhost:8001/health`
  - View logs for loading errors
- **Slow inference**: First request may be slower (JIT compilation)
- **OOM errors**: Reduce batch size or increase memory

## API Testing

### Bruno Collection
Import `bruno-collection/` into [Bruno](https://www.usebruno.com/) for interactive API testing:
- Health Check
- Single Text Embedding
- Batch Embeddings (5 texts)
- German Tax Law (10 multilingual texts)

### curl Examples
```bash
# Health check
curl http://localhost:8001/health

# Single embedding
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world"]}'

# Batch embeddings
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2", "Text 3"]}'
```

## Running with Reranker Service

Both services can run simultaneously:

```bash
# Terminal 1: Reranker (port 8000)
cd reranker-service
make run

# Terminal 2: Embeddings (port 8001)
cd embedding-service
make run

# Use together for two-stage retrieval
```

## Migration from OpenAI ada-002

BGE-M3 advantages over OpenAI text-embedding-ada-002:
- ✅ **Better multilingual performance** (especially German)
- ✅ **Self-hosted** (data privacy, no API costs)
- ✅ **Offline operation** (no internet required)
- ✅ **No rate limits** (process millions of texts)
- ✅ **Multi-functionality** (dense + sparse + multi-vector)
- ✅ **Longer context** (8192 vs 8191 tokens)

See [BGE-M3 vs ada-002 comparison](../docs/bge-m3-vs-ada002.md) for benchmarks.

## License

This service uses:
- **FlagEmbedding**: MIT License
- **BAAI/bge-m3**: MIT License
- **FastAPI**: MIT License

## Credits

- Model: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- Library: [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- Paper: [BGE M3-Embedding](https://arxiv.org/abs/2402.03216)
