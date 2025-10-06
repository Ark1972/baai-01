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
	docker build -f Dockerfile -t $(IMAGE_NAME):$(IMAGE_TAG) .

run: ## Run using docker-compose
	docker-compose up -d

stop: ## Stop docker-compose services
	docker-compose down

logs: ## View docker-compose logs
	docker-compose logs -f

test-local: ## Test the API locally
	@echo "Testing health endpoint..."
	curl -f http://localhost:$(PORT)/health || echo "Health check failed"
	@echo "\nTesting single rerank..."
	curl -X POST http://localhost:$(PORT)/rerank \
		-H "Content-Type: application/json" \
		-d '{"query": "test query", "passage": "test passage", "normalize": false}'

test-client: ## Run the Python client test
	python client/test_client.py

install-deps: ## Install Python dependencies locally
	pip install -r requirements.txt

clean: ## Clean up containers and images
	docker-compose down -v || true
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) || true

# Azure deployment commands
azure-login: ## Login to Azure
	az login

azure-acr-build: ## Build image in Azure Container Registry
	az acr build --registry $(ACR_NAME) --image $(IMAGE_NAME):$(IMAGE_TAG) --file Dockerfile .

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

# Documentation
docs: ## Show API documentation info
	@echo "API documentation available at http://localhost:$(PORT)/docs"
	@echo "ReDoc available at http://localhost:$(PORT)/redoc"
