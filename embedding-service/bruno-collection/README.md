# BAAI BGE-M3 Embedding Service - Bruno API Collection

This Bruno collection contains all API endpoints for testing the BAAI BGE-M3 Embedding Service.

## Setup

1. Install [Bruno](https://www.usebruno.com/)
2. Open Bruno and import this collection folder
3. Select the "local" environment (localhost:8001)
4. Ensure the service is running: `cd embedding-service && make run`

## Endpoints

### 1. Health Check
- **Method**: GET
- **Path**: `/health`
- **Purpose**: Verify service and model status
- **Response**: Model info, device (cpu/cuda), version, dimensions

### 2. Generate Embeddings - Single
- **Method**: POST
- **Path**: `/embed`
- **Purpose**: Generate embedding for single text
- **Request**: `{ texts: ["text to embed"] }`
- **Response**: `{ embeddings: [[1024 floats]], dimensions: 1024, texts_count: 1 }`

### 3. Generate Embeddings - Batch
- **Method**: POST
- **Path**: `/embed`
- **Purpose**: Generate embeddings for multiple texts
- **Request**: `{ texts: ["text1", "text2", ...] }`
- **Response**: `{ embeddings: [[...], [...]], dimensions: 1024, texts_count: N }`

### 4. Generate Embeddings - German Tax Law
- **Method**: POST
- **Path**: `/embed`
- **Purpose**: Test multilingual capabilities with German legal texts
- **Texts**: 10 German tax law passages
- **Use case**: Demonstrates BGE-M3's strong multilingual performance

## Environment Variables

The collection uses the following variable:
- `baseUrl`: Default `http://localhost:8001`

You can modify this in `environments/local.bru` or create additional environments.

## Testing Workflow

1. **Start Service**: `cd embedding-service && make run`
2. **Health Check**: Verify model is loaded
3. **Single Embedding**: Test with one text
4. **Batch Embeddings**: Test with 5 texts
5. **German Tax Law**: Test multilingual capabilities with 10 German texts

## Model Features

- **Model**: BAAI/bge-m3
- **Dimensions**: 1024
- **Languages**: 100+
- **Max Input**: 8192 tokens (~50000 characters)
- **Batch Size**: 1-100 texts
- **Multi-Functionality**: Dense retrieval (sparse/multi-vector also supported)

## API Documentation

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **OpenAPI JSON**: http://localhost:8001/openapi.json

## Notes

- The service pre-downloads the model during Docker build (~2GB)
- Startup time: 15-30 seconds (model loading from cache)
- CPU inference is supported (GPU optional via `USE_FP16=true`)
- BGE-M3 excels at multilingual tasks, especially non-English languages

## Integration with Reranker Service

BGE-M3 Embedding Service (port 8001) can be used with the Reranker Service (port 8000)
for a complete two-stage retrieval pipeline:

1. **Stage 1**: Generate embeddings → Search vector DB → Get top-100 candidates
2. **Stage 2**: Rerank top-100 → Get top-20 most relevant results

Both services can run simultaneously on different ports.
