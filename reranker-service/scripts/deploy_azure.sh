#!/bin/bash

# Azure Deployment Script for BAAI Reranker Service
# This script automates the deployment process to Azure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-reranker-service}"
LOCATION="${LOCATION:-eastus}"
ACR_NAME="${ACR_NAME:-acrrerankerservice}"
CONTAINER_NAME="${CONTAINER_NAME:-aci-reranker-service}"
IMAGE_NAME="${IMAGE_NAME:-reranker-service}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo -e "${GREEN}🚀 BAAI Reranker Service - Azure Deployment Script${NC}"
echo "=================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "\n${YELLOW}📋 Checking prerequisites...${NC}"

if ! command_exists az; then
    echo -e "${RED}❌ Azure CLI is not installed. Please install it first.${NC}"
    echo "Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

if ! command_exists docker; then
    echo -e "${RED}❌ Docker is not installed. Please install it first.${NC}"
    exit 1
fi

if ! command_exists terraform; then
    echo -e "${YELLOW}⚠️  Terraform is not installed. Will use Azure CLI instead.${NC}"
    USE_TERRAFORM=false
else
    USE_TERRAFORM=true
fi

# Login to Azure
echo -e "\n${YELLOW}🔐 Logging into Azure...${NC}"
az account show >/dev/null 2>&1 || az login

# Get subscription ID
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo -e "${GREEN}✅ Using subscription: ${SUBSCRIPTION_ID}${NC}"

# Build Docker image
echo -e "\n${YELLOW}🔨 Building Docker image...${NC}"
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Failed to build Docker image${NC}"
    exit 1
fi

# Create resource group
echo -e "\n${YELLOW}📁 Creating resource group...${NC}"
az group create --name ${RESOURCE_GROUP} --location ${LOCATION} >/dev/null

# Create Azure Container Registry
echo -e "\n${YELLOW}🏗️  Creating Azure Container Registry...${NC}"
az acr create --resource-group ${RESOURCE_GROUP} \
    --name ${ACR_NAME} \
    --sku Basic \
    --admin-enabled true >/dev/null

echo -e "${GREEN}✅ ACR created: ${ACR_NAME}${NC}"

# Get ACR credentials
echo -e "\n${YELLOW}🔑 Getting ACR credentials...${NC}"
ACR_LOGIN_SERVER=$(az acr show --name ${ACR_NAME} --query loginServer -o tsv)
ACR_USERNAME=$(az acr credential show --name ${ACR_NAME} --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name ${ACR_NAME} --query passwords[0].value -o tsv)

# Login to ACR
echo -e "\n${YELLOW}🔐 Logging into ACR...${NC}"
echo ${ACR_PASSWORD} | docker login ${ACR_LOGIN_SERVER} -u ${ACR_USERNAME} --password-stdin

# Tag and push image
echo -e "\n${YELLOW}📤 Pushing image to ACR...${NC}"
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}
docker push ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Image pushed to ACR successfully${NC}"
else
    echo -e "${RED}❌ Failed to push image to ACR${NC}"
    exit 1
fi

# Deploy using Terraform or Azure CLI
if [ "$USE_TERRAFORM" = true ]; then
    echo -e "\n${YELLOW}🏗️  Deploying with Terraform...${NC}"
    
    cd terraform
    
    # Create terraform.tfvars
    cat > terraform.tfvars <<EOF
subscription_id = "${SUBSCRIPTION_ID}"
resource_group_name = "${RESOURCE_GROUP}"
location = "${LOCATION}"
container_registry_name = "${ACR_NAME}"
container_instance_name = "${CONTAINER_NAME}"
container_image_name = "${IMAGE_NAME}"
container_image_tag = "${IMAGE_TAG}"
EOF
    
    terraform init
    terraform apply -auto-approve
    
    # Get outputs
    API_ENDPOINT=$(terraform output -raw api_endpoint)
    API_DOCS_URL=$(terraform output -raw api_docs_url)
    
    cd ..
else
    echo -e "\n${YELLOW}🏗️  Deploying with Azure CLI...${NC}"
    
    # Create storage account for model cache (optional)
    STORAGE_ACCOUNT="sarerankercache"
    echo -e "${YELLOW}Creating storage account...${NC}"
    az storage account create \
        --name ${STORAGE_ACCOUNT} \
        --resource-group ${RESOURCE_GROUP} \
        --location ${LOCATION} \
        --sku Standard_LRS >/dev/null
    
    STORAGE_KEY=$(az storage account keys list \
        --account-name ${STORAGE_ACCOUNT} \
        --resource-group ${RESOURCE_GROUP} \
        --query "[0].value" -o tsv)
    
    # Create file share
    az storage share create \
        --name model-cache \
        --account-name ${STORAGE_ACCOUNT} \
        --account-key ${STORAGE_KEY} \
        --quota 50 >/dev/null
    
    # Deploy container instance
    echo -e "${YELLOW}Deploying container instance...${NC}"
    az container create \
        --resource-group ${RESOURCE_GROUP} \
        --name ${CONTAINER_NAME} \
        --image ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG} \
        --cpu 2 \
        --memory 4 \
        --registry-login-server ${ACR_LOGIN_SERVER} \
        --registry-username ${ACR_USERNAME} \
        --registry-password ${ACR_PASSWORD} \
        --dns-name-label reranker-service-${RANDOM} \
        --ports 8000 \
        --environment-variables \
            MODEL_NAME=BAAI/bge-reranker-v2-m3 \
            USE_FP16=true \
            MODEL_CACHE_DIR=/app/models \
            PORT=8000 \
            HOST=0.0.0.0 \
            CORS_ORIGINS="*" \
        --azure-file-volume-account-name ${STORAGE_ACCOUNT} \
        --azure-file-volume-account-key ${STORAGE_KEY} \
        --azure-file-volume-share-name model-cache \
        --azure-file-volume-mount-path /app/models
    
    # Get container details
    FQDN=$(az container show \
        --resource-group ${RESOURCE_GROUP} \
        --name ${CONTAINER_NAME} \
        --query ipAddress.fqdn -o tsv)
    
    API_ENDPOINT="http://${FQDN}:8000"
    API_DOCS_URL="http://${FQDN}:8000/docs"
fi

# Wait for container to be ready
echo -e "\n${YELLOW}⏳ Waiting for container to be ready...${NC}"
sleep 30

# Test the deployment
echo -e "\n${YELLOW}🧪 Testing deployment...${NC}"
if curl -f ${API_ENDPOINT}/health >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Health check passed!${NC}"
else
    echo -e "${YELLOW}⚠️  Container might still be starting up...${NC}"
fi

# Display deployment information
echo -e "\n${GREEN}🎉 Deployment Complete!${NC}"
echo "======================================"
echo -e "API Endpoint: ${GREEN}${API_ENDPOINT}${NC}"
echo -e "API Documentation: ${GREEN}${API_DOCS_URL}${NC}"
echo -e "Resource Group: ${RESOURCE_GROUP}"
echo -e "Container Registry: ${ACR_NAME}"
echo -e "Container Instance: ${CONTAINER_NAME}"
echo ""
echo "To view logs:"
echo "  az container logs --resource-group ${RESOURCE_GROUP} --name ${CONTAINER_NAME}"
echo ""
echo "To delete resources:"
echo "  az group delete --name ${RESOURCE_GROUP} --yes"