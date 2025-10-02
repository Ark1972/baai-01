# BAAI Reranker Service

A containerized web service for text reranking using BAAI's FlagEmbedding model (bge-reranker-v2-m3). This service provides a RESTful API for reranking query-passage pairs, optimized for deployment on Azure.

## Features

- **FastAPI-based REST API** with automatic documentation
- **Single and batch reranking** endpoints
- **Score normalization** (0-1 range using sigmoid)
- **Multi-stage Docker build** for optimized image size
- **Azure deployment ready** with Terraform infrastructure as code
- **Health monitoring** endpoints
- **CORS support** for web applications
- **Model caching** to reduce startup time
- **Comprehensive logging** and error handling

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)
- Azure CLI (for Azure deployment)
- Terraform (optional, for infrastructure deployment)

### Local Development

1. **Clone and navigate to the project:**
```bash
cd reranker-service
```

2. **Build and run with Docker Compose:**
```bash
docker-compose up -d
```

3. **Test the service:**
```bash
# Health check
curl http://localhost:8000/health

# Single rerank
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "passage": "Machine learning is a subset of AI...",
    "normalize": false
  }'
```

4. **View API documentation:**
- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health Check
```http
GET /health
```
Returns service status and model information.

### Single Reranking
```http
POST /rerank
```
Rerank a single query-passage pair.

**Request Body:**
```json
{
  "query": "Your query text",
  "passage": "Passage to rank",
  "normalize": false
}
```

**Response:**
```json
{
  "score": -5.65234375,
  "normalized": false,
  "query_length": 15,
  "passage_length": 20
}
```

### Batch Reranking
```http
POST /rerank/batch
```
Rerank multiple query-passage pairs.

**Request Body:**
```json
{
  "pairs": [
    {"query": "Query 1", "passage": "Passage 1"},
    {"query": "Query 2", "passage": "Passage 2"}
  ],
  "normalize": true
}
```

**Response:**
```json
{
  "scores": [0.9948, 0.0027],
  "normalized": true,
  "pairs_count": 2
}
```

## Deployment

### Using Makefile

```bash
# Build Docker image
make build

# Run locally
make run

# Run tests
make test-local

# Deploy to Azure (full pipeline)
make deploy

# View all commands
make help
```

### Manual Docker Build

```bash
# Build image
docker build -t reranker-service:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e MODEL_NAME=BAAI/bge-reranker-v2-m3 \
  -e USE_FP16=true \
  reranker-service:latest
```

### Azure Deployment

#### Option 1: Using Deployment Script

```bash
# Make script executable
chmod +x scripts/deploy_azure.sh

# Run deployment
./scripts/deploy_azure.sh
```

#### Option 2: Using Terraform

1. **Configure variables:**
```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your Azure subscription ID
```

2. **Deploy infrastructure:**
```bash
terraform init
terraform plan
terraform apply
```

3. **Get deployment outputs:**
```bash
terraform output api_endpoint
```

#### Option 3: Manual Azure CLI

```bash
# Set variables
RESOURCE_GROUP="rg-reranker-service"
ACR_NAME="acrrerankerservice"
LOCATION="eastus"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create container registry
az acr create --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME --sku Basic --admin-enabled true

# Build and push image
az acr build --registry $ACR_NAME \
  --image reranker-service:latest .

# Deploy container instance
az container create \
  --resource-group $RESOURCE_GROUP \
  --name aci-reranker-service \
  --image $ACR_NAME.azurecr.io/reranker-service:latest \
  --cpu 2 --memory 4 \
  --registry-login-server $ACR_NAME.azurecr.io \
  --registry-username $(az acr credential show --name $ACR_NAME --query username -o tsv) \
  --registry-password $(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv) \
  --dns-name-label reranker-service \
  --ports 8000
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | BAAI model to use | `BAAI/bge-reranker-v2-m3` |
| `USE_FP16` | Use FP16 precision | `true` |
| `MODEL_CACHE_DIR` | Directory for model cache | `/app/models` |
| `PORT` | Service port | `8000` |
| `HOST` | Service host | `0.0.0.0` |
| `CORS_ORIGINS` | CORS allowed origins | `*` |

### Terraform Variables

See `terraform/terraform.tfvars.example` for all configurable options including:
- Azure subscription and resource group
- Container resources (CPU, memory)
- Storage configuration
- Monitoring settings

## Development

### Project Structure
```
reranker-service/
├── app/
│   └── main.py              # FastAPI application
├── client/
│   └── reranker_client.py   # Python client example
├── terraform/
│   ├── main.tf              # Infrastructure definition
│   ├── variables.tf         # Variable declarations
│   └── outputs.tf           # Output definitions
├── scripts/
│   └── deploy_azure.sh      # Deployment automation
├── tests/
│   ├── test_api.py          # Unit tests
│   └── performance_test.py  # Performance tests
├── Dockerfile               # Multi-stage Docker build
├── docker-compose.yml       # Local development setup
├── requirements.txt         # Python dependencies
└── Makefile                # Build automation
```

### Running Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run unit tests
pytest tests/ -v

# Run performance tests
python tests/performance_test.py

# Test with client
python client/reranker_client.py
```

### Local Development without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Performance Considerations

1. **FP16 Precision**: Enabled by default for faster inference with minimal accuracy loss
2. **Model Caching**: Models are cached to persistent storage to avoid re-downloading
3. **Batch Processing**: Use batch endpoint for multiple pairs (up to 100) for better throughput
4. **Resource Allocation**: Recommended minimum: 2 CPU cores, 4GB RAM

## Monitoring

- **Health Endpoint**: `/health` for container health checks
- **Application Insights**: Optional Azure monitoring integration
- **Container Logs**: Available via Docker or Azure CLI
- **Metrics**: Request count, latency, and error rates

## Troubleshooting

### Container fails to start
- Check memory allocation (minimum 4GB recommended)
- Verify model download completed
- Check logs: `docker logs <container-name>`

### Slow first request
- Model loading takes 30-60 seconds on first startup
- Use persistent volume for model cache
- Consider pre-downloading model in Docker image

### API errors
- Check request format matches API specification
- Verify text length limits (max 10,000 characters)
- Check CORS settings for web clients

## Security Considerations

- Use HTTPS in production (configure via reverse proxy)
- Restrict CORS origins in production
- Use Azure Key Vault for sensitive configuration
- Enable authentication/authorization as needed
- Regular security updates for base images

## License

This project uses BAAI's bge-reranker-v2-m3 model. Please refer to the model's license for usage terms.

## Support

For issues, questions, or contributions:
1. Check the API documentation at `/docs`
2. Review logs for error details
3. Test with the provided Python client
4. Check Azure resource status if deployed

## Acknowledgments

- [BAAI](https://github.com/FlagOpen/FlagEmbedding) for the excellent reranker model
- FastAPI for the modern web framework
- Azure for cloud infrastructure