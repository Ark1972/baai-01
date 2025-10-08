.PHONY: build run stop logs test-local dev install-deps clean

# Build the Docker image
build:
	@echo "Building BAAI BGE-M3 Embedding Service Docker image..."
	docker-compose build

# Start the service
run:
	@echo "Starting BAAI BGE-M3 Embedding Service..."
	docker-compose up -d
	@echo "Service starting on http://localhost:8001"
	@echo "Swagger UI: http://localhost:8001/docs"

# Stop the service
stop:
	@echo "Stopping BAAI BGE-M3 Embedding Service..."
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Test the service locally
test-local:
	@echo "Testing embedding service endpoints..."
	@echo "\n1. Health Check:"
	curl -s http://localhost:8001/health | python -m json.tool || echo "Service not ready"
	@echo "\n2. Root endpoint:"
	curl -s http://localhost:8001/ | python -m json.tool || echo "Service not ready"
	@echo "\n3. Embedding generation (single text):"
	curl -s -X POST http://localhost:8001/embed \
		-H "Content-Type: application/json" \
		-d '{"texts": ["This is a test sentence for embedding generation."]}' \
		| python -m json.tool || echo "Service not ready"

# Run test client
test-client:
	python client/test_client.py

# Development mode (run locally without Docker)
dev:
	@echo "Starting embedding service in development mode..."
	cd app && uvicorn main:app --reload --host 0.0.0.0 --port 8001

# Install Python dependencies locally
install-deps:
	pip install -r requirements.txt

# Clean up Docker resources
clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v
	docker system prune -f
