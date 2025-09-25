#!/bin/bash

# Azure Deployment Script for Agentic AI Blueprint Analyzer
# This script deploys the application to Azure App Service

set -e

# Configuration
RESOURCE_GROUP=${RESOURCE_GROUP:-"rg-agentic-blueprint"}
LOCATION=${LOCATION:-"eastus"}
ACR_NAME=${ACR_NAME:-"acragenticblueprint"}
APP_PLAN=${APP_PLAN:-"plan-agentic-blueprint"}
APP_NAME=${APP_NAME:-"app-agentic-blueprint"}
IMAGE_NAME="agentic-blueprint-app"
IMAGE_TAG=${IMAGE_TAG:-"latest"}

echo "🚀 Starting Azure deployment..."

# Check if logged in to Azure
echo "📝 Checking Azure login..."
az account show > /dev/null 2>&1 || az login

# Create resource group
echo "📦 Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
echo "🐳 Creating Azure Container Registry..."
az acr create --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --admin-enabled true

# Build and push Docker image
echo "🔨 Building Docker image..."
az acr build --registry $ACR_NAME \
  --image $IMAGE_NAME:$IMAGE_TAG \
  --file Dockerfile .

# Create App Service Plan
echo "📋 Creating App Service Plan..."
az appservice plan create --name $APP_PLAN \
  --resource-group $RESOURCE_GROUP \
  --is-linux \
  --sku B2

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)

# Create Web App
echo "🌐 Creating Web App..."
az webapp create --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --plan $APP_PLAN \
  --deployment-container-image-name $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG

# Configure container settings
echo "⚙️ Configuring container settings..."
az webapp config container set --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --docker-custom-image-name $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG \
  --docker-registry-server-url https://$ACR_NAME.azurecr.io \
  --docker-registry-server-user $ACR_USERNAME \
  --docker-registry-server-password $ACR_PASSWORD

# Configure app settings
echo "🔧 Configuring app settings..."
az webapp config appsettings set --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings \
    WEBSITES_PORT=8000 \
    AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT}" \
    AZURE_OPENAI_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-gpt-4}" \
    AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-02-15-preview}" \
    ALLOWED_ORIGINS="https://$APP_NAME.azurewebsites.net" \
    ENV="production"

# Enable managed identity
echo "🔐 Enabling managed identity..."
az webapp identity assign --name $APP_NAME \
  --resource-group $RESOURCE_GROUP

# Get the app URL
APP_URL=$(az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query defaultHostName -o tsv)

echo "✅ Deployment complete!"
echo "🌐 Application URL: https://$APP_URL"
echo ""
echo "⚠️ Next steps:"
echo "1. Grant the managed identity access to your Azure OpenAI resource"
echo "2. Configure any additional environment variables in the Azure Portal"
echo "3. Set up continuous deployment if desired"