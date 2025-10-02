variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
  default     = "rg-reranker-service"
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

variable "container_registry_name" {
  description = "Name of the Azure Container Registry"
  type        = string
  default     = "acrrerankerservice"
}

variable "container_registry_sku" {
  description = "SKU of the Azure Container Registry"
  type        = string
  default     = "Basic"
}

variable "container_instance_name" {
  description = "Name of the Azure Container Instance"
  type        = string
  default     = "aci-reranker-service"
}

variable "container_image_name" {
  description = "Name of the container image"
  type        = string
  default     = "reranker-service"
}

variable "container_image_tag" {
  description = "Tag of the container image"
  type        = string
  default     = "latest"
}

variable "container_cpu" {
  description = "Number of CPU cores for the container"
  type        = number
  default     = 2
}

variable "container_memory" {
  description = "Memory in GB for the container"
  type        = number
  default     = 4
}

variable "dns_name_label" {
  description = "DNS name label for the container instance"
  type        = string
  default     = "reranker-service"
}

variable "restart_policy" {
  description = "Restart policy for the container"
  type        = string
  default     = "OnFailure"
}

variable "model_name" {
  description = "Name of the BAAI model to use"
  type        = string
  default     = "BAAI/bge-reranker-v2-m3"
}

variable "use_fp16" {
  description = "Whether to use FP16 precision"
  type        = bool
  default     = true
}

variable "cors_origins" {
  description = "CORS origins for the API"
  type        = string
  default     = "*"
}

variable "enable_model_cache_storage" {
  description = "Enable Azure File Share for model cache"
  type        = bool
  default     = true
}

variable "storage_account_name" {
  description = "Name of the storage account for model cache"
  type        = string
  default     = "sarerankercache"
}

variable "enable_application_insights" {
  description = "Enable Application Insights for monitoring"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Environment = "Production"
    Service     = "RerankerService"
    ManagedBy   = "Terraform"
  }
}