# 🤖 Agentic AI Blueprint Analyzer

A production-grade full-stack application that analyzes business use cases and generates visual blueprints showing which agentic AI components are required to solve them.

## 🚀 Features

- **AI-Powered Analysis**: Uses Azure OpenAI GPT-4 to analyze business use cases
- **Interactive Visualization**: Dynamic blueprint that highlights required components
- **Enterprise Security**: Azure AD authentication via DefaultAzureCredential
- **Production Ready**: Dockerized, scalable, and deployable to Azure App Service
- **Real-time Feedback**: Animated component highlighting based on analysis results

## 🏗️ Architecture

### Tech Stack
- **Backend**: Python FastAPI
- **Frontend**: Vue 3 + Vite
- **LLM**: Azure OpenAI (GPT-4)
- **Authentication**: Azure Entra ID / MSAL
- **Deployment**: Azure App Service compatible

### Components Analyzed
- **Reasoning Engine**: LangGraph for complex workflows
- **Memory Management**: Redis for session/long-term memory
- **RAG**: Retrieval-Augmented Generation for internal data
- **Evaluation**: RAGAS for output quality measurement
- **MCP Integration**: Connect to internal systems (CRM, banking, etc.)
- **Observability**: Always-on logging, tracing, and metrics
- **Agents**: Domain-specific agents (A, B, C)

## 📋 Prerequisites

- Python 3.11+
- Node.js 20+
- Azure subscription with OpenAI service
- Azure AD app registration (for production)

## 🔧 Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd agentic-blueprint-app
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

### 4. Environment Configuration

Create a `.env` file in the backend directory:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:5173

# Application Configuration
PORT=8000
ENV=development
DEBUG=false
```

## 🚀 Running Locally

### Option 1: Run separately

**Backend:**
```bash
cd backend
python main.py
```

**Frontend:**
```bash
cd frontend
npm run dev
```

### Option 2: Using Docker Compose

```bash
docker-compose up
```

Access the application at `http://localhost:5173`

## 🧪 Testing the Application

1. Open the application in your browser
2. Enter a business use case, for example:
   - "I need a customer service chatbot that can access our CRM system and remember past conversations"
   - "Build an AI system to analyze financial documents and provide investment recommendations"
   - "Create an automated code review system that learns from our codebase"

3. Click "Analyze Use Case"
4. Watch as the blueprint highlights the required components with animations

## 📦 Deployment

### Azure App Service

1. Build the Docker image:
```bash
docker build -t agentic-blueprint-app .
```

2. Push to Azure Container Registry:
```bash
az acr build --registry <your-registry> --image agentic-blueprint-app .
```

3. Deploy to App Service:
```bash
az webapp create --name <app-name> --resource-group <rg-name> --plan <plan-name> --deployment-container-image-name <registry>.azurecr.io/agentic-blueprint-app:latest
```

4. Configure environment variables in App Service:
```bash
az webapp config appsettings set --name <app-name> --resource-group <rg-name> --settings AZURE_OPENAI_ENDPOINT="<endpoint>" AZURE_OPENAI_DEPLOYMENT="gpt-4"
```

### Local Docker Build

```bash
# Build production image
docker build -t agentic-blueprint-app .

# Run production container
docker run -p 8000:8000 --env-file backend/.env agentic-blueprint-app
```

## 🔐 Security Considerations

- **No Secrets in Code**: Uses DefaultAzureCredential for authentication
- **CORS Protection**: Configured allowed origins
- **Input Validation**: Pydantic models validate all inputs
- **Error Handling**: Comprehensive error handling without exposing sensitive data
- **HTTPS Only**: Enforce HTTPS in production

## 📊 API Documentation

### POST /api/analyze-usecase

Analyzes a business use case and returns required AI components.

**Request:**
```json
{
  "usecase": "Your business use case description here"
}
```

**Response:**
```json
{
  "reasoning_engine": true,
  "memory": true,
  "rag": false,
  "evaluation": false,
  "mcp_integration": true,
  "observability": true,
  "simple_direct_answer": false,
  "agents": ["agent_a", "agent_c"]
}
```

### GET /api/health

Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "azure_openai_configured": true,
  "version": "1.0.0"
}
```

## 🎨 Frontend Components

The frontend visualizes the AI architecture blueprint with:
- **Input Layer**: User request entry point
- **Decision Layer**: Complexity analysis
- **Processing Layer**: LangGraph reasoning or direct answers
- **Agents Layer**: Domain-specific agents
- **Services Layer**: Memory, RAG, MCP, Evaluation
- **Output Layer**: Final response
- **Observability Layer**: Always-active monitoring

Components light up with animations when they're required for the analyzed use case.

## 🛠️ Development

### Adding New Components

1. Update the system prompt in `backend/main.py`
2. Add the component to the `ComponentAnalysis` model
3. Add visual representation in `frontend/src/App.vue`
4. Update styles in `frontend/src/style.scss`

### Extending the LLM Logic

Modify the `SYSTEM_PROMPT` in `backend/main.py` to change how use cases are analyzed.

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📞 Support

For issues and questions, please open an issue in the repository.

---

Built with ❤️ for enterprise-grade agentic AI solutions