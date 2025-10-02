output "resource_group_name" {
  description = "Name of the created resource group"
  value       = azurerm_resource_group.main.name
}

output "container_registry_login_server" {
  description = "Login server for the container registry"
  value       = azurerm_container_registry.main.login_server
}

output "container_registry_admin_username" {
  description = "Admin username for the container registry"
  value       = azurerm_container_registry.main.admin_username
  sensitive   = true
}

output "container_registry_admin_password" {
  description = "Admin password for the container registry"
  value       = azurerm_container_registry.main.admin_password
  sensitive   = true
}

output "container_instance_fqdn" {
  description = "Fully qualified domain name of the container instance"
  value       = azurerm_container_group.main.fqdn
}

output "container_instance_ip" {
  description = "Public IP address of the container instance"
  value       = azurerm_container_group.main.ip_address
}

output "api_endpoint" {
  description = "API endpoint URL"
  value       = "http://${azurerm_container_group.main.fqdn}:8000"
}

output "api_docs_url" {
  description = "API documentation URL"
  value       = "http://${azurerm_container_group.main.fqdn}:8000/docs"
}

output "storage_account_name" {
  description = "Name of the storage account (if enabled)"
  value       = var.enable_model_cache_storage ? azurerm_storage_account.model_cache[0].name : null
}

output "application_insights_instrumentation_key" {
  description = "Application Insights instrumentation key (if enabled)"
  value       = var.enable_application_insights ? azurerm_application_insights.main[0].instrumentation_key : null
  sensitive   = true
}