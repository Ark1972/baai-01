# BAAI Reranker Service - Bruno API Collection

This Bruno collection contains all API endpoints for testing the BAAI Reranker Service.

## Setup

1. Install [Bruno](https://www.usebruno.com/)
2. Open Bruno and import this collection folder
3. Select the "local" environment (localhost:8000)
4. Ensure the service is running: `cd reranker-service && make run`

## Endpoints

### 1. Health Check
- **Method**: GET
- **Path**: `/health`
- **Purpose**: Verify service and model status
- **Response**: Model info, device (cpu/cuda), version

### 2. Single Rerank
- **Method**: POST
- **Path**: `/rerank`
- **Purpose**: Rerank a single query-passage pair
- **Request**: `{ query, passage, normalize? }`
- **Response**: `{ score, normalized, query_length, passage_length }`
- **Default**: `normalize=true` (0-1 range using sigmoid)

### 3. Batch Rerank
- **Method**: POST
- **Path**: `/rerank/batch`
- **Purpose**: Rerank multiple query-passage pairs
- **Request**: `{ pairs: [{ query, passage }], normalize? }`
- **Response**: `{ scores: [], normalized, pairs_count }`
- **Optimization**: Groups pairs by same query for efficiency

### 4. Query Rerank
- **Method**: POST
- **Path**: `/rerank/query`
- **Purpose**: Rerank passages for a query, sorted by relevance
- **Request**: `{ query, passages: [], normalize? }`
- **Response**: `{ re_ranked: [{ passage, score }] }`
- **Sorting**: Results sorted by score descending (most relevant first)

### 5. Query Rerank - German Tax Law (20 passages)
- **Method**: POST
- **Path**: `/rerank/query`
- **Purpose**: Test with 20 highly relevant German tax law passages
- **Query**: "Welche Steuervorteile gelten für Selbständige bei der Abschreibung von Betriebsmitteln?"
- **Expected**: All passages should score 0.90+ due to high relevance

## Environment Variables

The collection uses the following variable:
- `baseUrl`: Default `http://localhost:8000`

You can modify this in `environments/local.bru` or create additional environments for staging/production.

## Testing Workflow

1. **Start Service**: `cd reranker-service && make run`
2. **Health Check**: Verify model is loaded
3. **Single Rerank**: Test basic functionality
4. **Batch Rerank**: Test multiple pairs
5. **Query Rerank**: Test sorted results (most common use case)
6. **German Tax Law**: Test with realistic, highly relevant data

## Features

- **Normalization**: All endpoints default to `normalize=true` (sigmoid 0-1 range)
- **Validation**: 1-10,000 chars per text, 1-100 items in arrays
- **Model**: BAAI/bge-reranker-v2-m3 (cross-encoder, multilingual)
- **Backend**: FlagEmbedding library (official BAAI implementation)

## API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Notes

- The service pre-downloads the model during Docker build (~2.3GB)
- Startup time: 15-30 seconds (model loading from cache)
- CPU inference is supported (GPU optional via `USE_FP16=true`)
- Cross-encoder model: processes [query, passage] jointly for accurate scoring
