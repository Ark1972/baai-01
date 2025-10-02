terraform {
  required_version = ">= 1.0"
  
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }

  # Optional: Configure backend for remote state storage
  # backend "azurerm" {
  #   resource_group_name  = "terraform-state-rg"
  #   storage_account_name = "tfstatestore"
  #   container_name       = "tfstate"
  #   key                  = "reranker-service.tfstate"
  # }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
  
  tags = var.tags
}

# Azure Container Registry
resource "azurerm_container_registry" "main" {
  name                = var.container_registry_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = var.container_registry_sku
  admin_enabled       = true

  tags = var.tags
}

# Storage Account for model cache (optional)
resource "azurerm_storage_account" "model_cache" {
  count = var.enable_model_cache_storage ? 1 : 0
  
  name                     = var.storage_account_name
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  
  tags = var.tags
}

# File Share for model cache
resource "azurerm_storage_share" "model_cache" {
  count = var.enable_model_cache_storage ? 1 : 0
  
  name                 = "model-cache"
  storage_account_name = azurerm_storage_account.model_cache[0].name
  quota                = 50
}

# Azure Container Instance
resource "azurerm_container_group" "main" {
  name                = var.container_instance_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  os_type             = "Linux"
  restart_policy      = var.restart_policy
  ip_address_type     = "Public"
  dns_name_label      = var.dns_name_label

  image_registry_credential {
    server   = azurerm_container_registry.main.login_server
    username = azurerm_container_registry.main.admin_username
    password = azurerm_container_registry.main.admin_password
  }

  container {
    name   = "reranker-service"
    image  = "${azurerm_container_registry.main.login_server}/${var.container_image_name}:${var.container_image_tag}"
    cpu    = var.container_cpu
    memory = var.container_memory

    ports {
      port     = 8000
      protocol = "TCP"
    }

    environment_variables = {
      MODEL_NAME      = var.model_name
      USE_FP16        = tostring(var.use_fp16)
      MODEL_CACHE_DIR = "/app/models"
      PORT            = "8000"
      HOST            = "0.0.0.0"
      CORS_ORIGINS    = var.cors_origins
    }

    # Optional: Mount Azure File Share for model cache
    dynamic "volume" {
      for_each = var.enable_model_cache_storage ? [1] : []
      content {
        name                 = "model-cache"
        mount_path           = "/app/models"
        share_name           = azurerm_storage_share.model_cache[0].name
        storage_account_name = azurerm_storage_account.model_cache[0].name
        storage_account_key  = azurerm_storage_account.model_cache[0].primary_access_key
      }
    }

    liveness_probe {
      http_get {
        path = "/health"
        port = 8000
      }
      initial_delay_seconds = 120
      period_seconds        = 30
      failure_threshold     = 3
      timeout_seconds       = 10
    }

    readiness_probe {
      http_get {
        path = "/health"
        port = 8000
      }
      initial_delay_seconds = 60
      period_seconds        = 10
      failure_threshold     = 3
      timeout_seconds       = 5
    }
  }

  tags = var.tags
}

# Application Insights for monitoring (optional)
resource "azurerm_application_insights" "main" {
  count = var.enable_application_insights ? 1 : 0
  
  name                = "${var.container_instance_name}-insights"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  application_type    = "web"
  
  tags = var.tags
}