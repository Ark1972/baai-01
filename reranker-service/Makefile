.PHONY: help build run stop test clean deploy

# Variables
IMAGE_NAME := reranker-service
IMAGE_TAG := latest
CONTAINER_NAME := baai-reranker-service
PORT := 8000
ACR_NAME := acrrerankerservice
RESOURCE_GROUP := rg-reranker-service

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

build: ## Build the Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

build-fast: ## Build using fast Dockerfile (pre-built ML base)
	docker build -f Dockerfile.fast -t $(IMAGE_NAME):fast .

run: ## Run the container locally
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):$(PORT) \
		-e MODEL_NAME=BAAI/bge-reranker-v2-m3 \
		-e USE_FP16=true \
		$(IMAGE_NAME):$(IMAGE_TAG)

run-compose: ## Run using docker-compose
	docker-compose up -d

run-compose-fast: ## Run using fast docker-compose
	docker-compose -f docker-compose.fast.yml up -d

stop: ## Stop the running container
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

stop-compose: ## Stop docker-compose services
	docker-compose down

logs: ## View container logs
	docker logs -f $(CONTAINER_NAME)

logs-compose: ## View docker-compose logs
	docker-compose logs -f

test-local: ## Test the API locally
	@echo "Testing health endpoint..."
	curl -f http://localhost:$(PORT)/health || echo "Health check failed"
	@echo "\nTesting single rerank..."
	curl -X POST http://localhost:$(PORT)/rerank \
		-H "Content-Type: application/json" \
		-d '{"query": "test query", "passage": "test passage", "normalize": false}'

test-client: ## Run the Python client test
	python client/reranker_client.py

install-deps: ## Install Python dependencies locally
	pip install -r requirements.txt

test-unit: ## Run unit tests
	pytest tests/ -v

test-performance: ## Run performance tests
	python tests/performance_test.py

clean: ## Clean up containers and images
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) || true
	docker-compose down -v || true

# Azure deployment commands
azure-login: ## Login to Azure
	az login

azure-acr-build: ## Build image in Azure Container Registry
	az acr build --registry $(ACR_NAME) --image $(IMAGE_NAME):$(IMAGE_TAG) .

azure-acr-push: build ## Push local image to Azure Container Registry
	@echo "Tagging image for ACR..."
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(ACR_NAME).azurecr.io/$(IMAGE_NAME):$(IMAGE_TAG)
	@echo "Logging into ACR..."
	az acr login --name $(ACR_NAME)
	@echo "Pushing image to ACR..."
	docker push $(ACR_NAME).azurecr.io/$(IMAGE_NAME):$(IMAGE_TAG)

# Terraform commands
tf-init: ## Initialize Terraform
	cd terraform && terraform init

tf-plan: ## Plan Terraform deployment
	cd terraform && terraform plan

tf-apply: ## Apply Terraform deployment
	cd terraform && terraform apply

tf-destroy: ## Destroy Terraform resources
	cd terraform && terraform destroy

tf-output: ## Show Terraform outputs
	cd terraform && terraform output

# Combined deployment
deploy: azure-acr-push tf-apply ## Full deployment to Azure

# Development commands
dev: ## Run in development mode with hot reload
	cd app && uvicorn main:app --reload --host 0.0.0.0 --port $(PORT)

format: ## Format Python code
	black app/ tests/ client/
	isort app/ tests/ client/

lint: ## Lint Python code
	flake8 app/ tests/ client/
	mypy app/ tests/ client/

security-scan: ## Run security scan on Docker image
	docker scan $(IMAGE_NAME):$(IMAGE_TAG)

# Documentation
docs: ## Generate API documentation
	@echo "API documentation available at http://localhost:$(PORT)/docs"
	@echo "ReDoc available at http://localhost:$(PORT)/redoc"