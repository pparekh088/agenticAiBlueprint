"""
FastAPI Backend for Agentic AI Blueprint Analyzer
Production-grade application using Azure OpenAI with DefaultAzureCredential
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

# System prompt for the LLM
SYSTEM_PROMPT = """You are an agentic AI solution analyst. You take a real-world use case and determine which components from an enterprise-grade agentic AI architecture are required to solve it.

## Component Definitions

- `reasoning_engine`: Use LangGraph for multi-step workflows or planning
- `memory`: Use Redis for session or long-term memory
- `rag`: Retrieval-Augmented Generation, if internal documents or knowledge bases are used
- `evaluation`: Use RAGAS or custom eval if output quality needs to be measured
- `mcp_integration`: Use MCP server to connect to internal systems (e.g., CRM, core banking)
- `observability`: Always true. Logs, traces, metrics must always be active
- `simple_direct_answer`: Set to true only if the use case is trivial and doesn't require other components
- `agents`: Array of domain-specific agents, e.g., `agent_a`, `agent_b`, `agent_c`

## Output Format (strict JSON):

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

Only include necessary components. Always include "observability": true.
Return ONLY the JSON object, no additional text or explanation."""

# Request/Response models
class UseCaseRequest(BaseModel):
    usecase: str = Field(..., min_length=10, max_length=5000, description="Business use case description")
    
    @validator('usecase')
    def validate_usecase(cls, v):
        if not v.strip():
            raise ValueError("Use case cannot be empty")
        return v.strip()

class ComponentAnalysis(BaseModel):
    reasoning_engine: bool = Field(default=False, description="Whether LangGraph reasoning is needed")
    memory: bool = Field(default=False, description="Whether Redis memory is needed")
    rag: bool = Field(default=False, description="Whether RAG is needed")
    evaluation: bool = Field(default=False, description="Whether RAGAS evaluation is needed")
    mcp_integration: bool = Field(default=False, description="Whether MCP integration is needed")
    observability: bool = Field(default=True, description="Observability is always enabled")
    simple_direct_answer: bool = Field(default=False, description="Whether it's a simple query")
    agents: List[str] = Field(default_factory=list, description="List of required agents")

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int

# Global Azure OpenAI client
azure_openai_client: Optional[AzureOpenAI] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    global azure_openai_client
    
    try:
        if not AZURE_OPENAI_ENDPOINT:
            logger.error("AZURE_OPENAI_ENDPOINT not configured")
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
        
        # Initialize Azure OpenAI client with DefaultAzureCredential
        credential = DefaultAzureCredential()
        azure_openai_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token
        )
        logger.info("Azure OpenAI client initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
        # Allow app to start but endpoints will return errors
        azure_openai_client = None
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")

# Create FastAPI app
app = FastAPI(
    title="Agentic AI Blueprint Analyzer",
    description="Analyzes business use cases and returns required AI components",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "azure_openai_configured": azure_openai_client is not None,
        "version": "1.0.0"
    }

@app.post("/analyze-usecase", response_model=ComponentAnalysis)
async def analyze_usecase(request: UseCaseRequest):
    """
    Analyze a business use case and return required AI components
    """
    if not azure_openai_client:
        logger.error("Azure OpenAI client not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is not available. Please check configuration."
        )
    
    try:
        logger.info(f"Analyzing use case: {request.usecase[:100]}...")
        
        # Call Azure OpenAI
        response = azure_openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this use case and return the required components: {request.usecase}"}
            ],
            temperature=0.1,  # Low temperature for consistent analysis
            max_tokens=500,
            response_format={"type": "json_object"}  # Ensure JSON response
        )
        
        # Extract and parse response
        response_text = response.choices[0].message.content
        logger.info(f"LLM Response: {response_text}")
        
        # Parse JSON response
        try:
            component_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {response_text}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Invalid response format from AI service: {str(e)}"
            )
        
        # Ensure observability is always true
        component_data["observability"] = True
        
        # Validate and return using Pydantic model
        return ComponentAnalysis(**component_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing use case: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze use case: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom exception handler for HTTPException"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="An unexpected error occurred",
            detail=str(exc) if os.getenv("DEBUG") == "true" else None,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        ).dict()
    )

if __name__ == "__main__":
    # Run with uvicorn for development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "development") == "development",
        log_level="info"
    )