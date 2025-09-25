#!/bin/bash

# Validation Script for Agentic AI Blueprint Analyzer
# Checks that all files and configurations are correct

set -e

echo "🔍 Validating Agentic AI Blueprint Analyzer"
echo "==========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track validation status
VALIDATION_PASSED=true

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅${NC} $2"
    else
        echo -e "${RED}❌${NC} $2 - File not found: $1"
        VALIDATION_PASSED=false
    fi
}

# Function to check directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✅${NC} $2"
    else
        echo -e "${RED}❌${NC} $2 - Directory not found: $1"
        VALIDATION_PASSED=false
    fi
}

# Function to check for string in file
check_contains() {
    if grep -q "$2" "$1" 2>/dev/null; then
        echo -e "${GREEN}✅${NC} $3"
    else
        echo -e "${RED}❌${NC} $3 - Missing in $1"
        VALIDATION_PASSED=false
    fi
}

echo "📁 Checking project structure..."
check_dir "backend" "Backend directory"
check_dir "frontend" "Frontend directory"
check_dir "frontend/src" "Frontend source directory"

echo ""
echo "📄 Checking backend files..."
check_file "backend/main.py" "Backend main application"
check_file "backend/requirements.txt" "Backend requirements"
check_file "backend/.env.example" "Backend environment example"

echo ""
echo "📄 Checking frontend files..."
check_file "frontend/package.json" "Frontend package.json"
check_file "frontend/vite.config.js" "Vite configuration"
check_file "frontend/index.html" "Frontend HTML entry"
check_file "frontend/src/main.js" "Frontend main.js"
check_file "frontend/src/App.vue" "Frontend App component"
check_file "frontend/src/style.scss" "Frontend styles"
check_file "frontend/src/config.js" "Frontend config"

echo ""
echo "🐳 Checking Docker files..."
check_file "Dockerfile" "Production Dockerfile"
check_file "Dockerfile.dev" "Development Dockerfile"
check_file "docker-compose.yml" "Docker Compose configuration"

echo ""
echo "📚 Checking documentation..."
check_file "README.md" "Project README"
check_file ".gitignore" "Git ignore file"

echo ""
echo "🔧 Checking deployment files..."
check_file "deploy-azure.sh" "Azure deployment script"
check_file "quickstart.sh" "Quick start script"

echo ""
echo "🔍 Validating backend code..."
# Check for critical imports
check_contains "backend/main.py" "from fastapi import" "FastAPI import"
check_contains "backend/main.py" "from azure.identity import DefaultAzureCredential" "Azure identity import"
check_contains "backend/main.py" "from openai import AzureOpenAI" "Azure OpenAI import"
check_contains "backend/main.py" "@app.post(\"/analyze-usecase\"" "Analyze endpoint"
check_contains "backend/main.py" "@app.get(\"/health\")" "Health endpoint"

echo ""
echo "🔍 Validating frontend code..."
check_contains "frontend/src/App.vue" "analyzeUseCase" "Analyze function"
check_contains "frontend/src/App.vue" "components.reasoning_engine" "Reasoning engine component"
check_contains "frontend/src/App.vue" "components.observability" "Observability component"
check_contains "frontend/package.json" "\"vue\":" "Vue dependency"
check_contains "frontend/package.json" "\"axios\":" "Axios dependency"

echo ""
echo "🔍 Checking configuration..."
check_contains "backend/.env.example" "AZURE_OPENAI_ENDPOINT" "Azure endpoint config"
check_contains "backend/.env.example" "AZURE_OPENAI_DEPLOYMENT" "Azure deployment config"
check_contains "frontend/vite.config.js" "proxy:" "Proxy configuration"

echo ""
echo "==========================================="
if [ "$VALIDATION_PASSED" = true ]; then
    echo -e "${GREEN}✅ All validations passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Copy backend/.env.example to backend/.env"
    echo "2. Add your Azure OpenAI credentials to backend/.env"
    echo "3. Run ./quickstart.sh to start the application"
    exit 0
else
    echo -e "${RED}❌ Some validations failed${NC}"
    echo ""
    echo "Please fix the issues above before running the application"
    exit 1
fi