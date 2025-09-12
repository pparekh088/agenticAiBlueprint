"""
MCP (Model Context Protocol) Server Implementation
Enterprise-grade MCP server for Financial Institutions
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import json
import logging
import jwt
from contextlib import asynccontextmanager
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

class ToolRegistration(BaseModel):
    """Tool registration request"""
    name: str
    description: str
    category: str
    version: str
    endpoint: str
    authentication_required: bool = False
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    rate_limits: Dict[str, int] = Field(default_factory=dict)
    permissions: List[str] = Field(default_factory=list)
    timeout: int = 30
    retry_count: int = 3


class ToolExecutionRequest(BaseModel):
    """Tool execution request"""
    tool_id: str
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    timeout_override: Optional[int] = None


class AgentRegistration(BaseModel):
    """Agent registration request"""
    name: str
    description: str
    capabilities: List[str]
    required_tools: List[str]
    permissions: List[str]
    metadata: Optional[Dict[str, Any]] = None


class TaskSubmission(BaseModel):
    """Task submission request"""
    task_type: str
    payload: Dict[str, Any]
    priority: str = "medium"
    dependencies: List[str] = Field(default_factory=list)
    deadline: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


# ============================================================================
# MCP Server Implementation
# ============================================================================

class MCPServer:
    """
    Model Context Protocol Server
    Manages tools, agents, and orchestration
    """
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.metrics = {
            "total_requests": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "active_sessions": 0
        }
    
    async def register_tool(self, tool: ToolRegistration, agent_id: str) -> Dict[str, Any]:
        """Register a new tool"""
        tool_id = f"tool_{len(self.tools) + 1:04d}"
        
        tool_data = {
            "id": tool_id,
            "name": tool.name,
            "description": tool.description,
            "category": tool.category,
            "version": tool.version,
            "endpoint": tool.endpoint,
            "authentication_required": tool.authentication_required,
            "input_schema": tool.input_schema,
            "output_schema": tool.output_schema,
            "rate_limits": tool.rate_limits,
            "permissions": tool.permissions,
            "timeout": tool.timeout,
            "retry_count": tool.retry_count,
            "registered_by": agent_id,
            "registered_at": datetime.now().isoformat(),
            "status": "active",
            "usage_count": 0
        }
        
        self.tools[tool_id] = tool_data
        logger.info(f"Registered tool: {tool.name} (ID: {tool_id})")
        
        return {
            "tool_id": tool_id,
            "status": "registered",
            "message": f"Tool {tool.name} registered successfully"
        }
    
    async def discover_tools(self, 
                           agent_id: str,
                           category: Optional[str] = None,
                           permissions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Discover available tools"""
        available_tools = []
        
        # Get agent permissions
        agent = self.agents.get(agent_id, {})
        agent_permissions = agent.get("permissions", [])
        
        for tool_id, tool in self.tools.items():
            # Filter by status
            if tool["status"] != "active":
                continue
            
            # Filter by category
            if category and tool["category"] != category:
                continue
            
            # Check permissions
            if permissions:
                tool_permissions = tool.get("permissions", [])
                if not all(p in agent_permissions for p in tool_permissions):
                    continue
            
            # Add to available tools
            available_tools.append({
                "id": tool_id,
                "name": tool["name"],
                "description": tool["description"],
                "category": tool["category"],
                "version": tool["version"],
                "permissions": tool["permissions"]
            })
        
        return available_tools
    
    async def execute_tool(self,
                         request: ToolExecutionRequest,
                         agent_id: str) -> Dict[str, Any]:
        """Execute a tool"""
        # Check if tool exists
        if request.tool_id not in self.tools:
            raise HTTPException(status_code=404, detail=f"Tool {request.tool_id} not found")
        
        tool = self.tools[request.tool_id]
        
        # Check permissions
        agent = self.agents.get(agent_id, {})
        agent_permissions = agent.get("permissions", [])
        tool_permissions = tool.get("permissions", [])
        
        if not all(p in agent_permissions for p in tool_permissions):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Update metrics
        self.metrics["total_requests"] += 1
        tool["usage_count"] += 1
        
        try:
            # Execute tool
            timeout = request.timeout_override or tool["timeout"]
            result = await self._execute_tool_call(
                tool["endpoint"],
                request.parameters,
                timeout
            )
            
            # Record execution
            execution_record = {
                "tool_id": request.tool_id,
                "agent_id": agent_id,
                "parameters": request.parameters,
                "result": result,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            self.execution_history.append(execution_record)
            
            # Update metrics
            self.metrics["successful_executions"] += 1
            
            return {
                "success": True,
                "result": result,
                "tool_id": request.tool_id,
                "execution_time": timeout,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            # Record failure
            execution_record = {
                "tool_id": request.tool_id,
                "agent_id": agent_id,
                "parameters": request.parameters,
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
            self.execution_history.append(execution_record)
            
            # Update metrics
            self.metrics["failed_executions"] += 1
            
            logger.error(f"Tool execution failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")
    
    async def _execute_tool_call(self,
                                endpoint: str,
                                parameters: Dict[str, Any],
                                timeout: int) -> Any:
        """Execute actual tool call"""
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, json=parameters)
            response.raise_for_status()
            return response.json()
    
    async def register_agent(self, agent: AgentRegistration) -> Dict[str, Any]:
        """Register a new agent"""
        agent_id = f"agent_{len(self.agents) + 1:04d}"
        
        agent_data = {
            "id": agent_id,
            "name": agent.name,
            "description": agent.description,
            "capabilities": agent.capabilities,
            "required_tools": agent.required_tools,
            "permissions": agent.permissions,
            "metadata": agent.metadata or {},
            "registered_at": datetime.now().isoformat(),
            "status": "active",
            "last_active": datetime.now().isoformat()
        }
        
        self.agents[agent_id] = agent_data
        logger.info(f"Registered agent: {agent.name} (ID: {agent_id})")
        
        return {
            "agent_id": agent_id,
            "status": "registered",
            "message": f"Agent {agent.name} registered successfully"
        }
    
    async def create_session(self, agent_id: str) -> Dict[str, Any]:
        """Create a new session for an agent"""
        if agent_id not in self.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{agent_id}"
        
        session_data = {
            "id": session_id,
            "agent_id": agent_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "status": "active",
            "context": {},
            "execution_count": 0
        }
        
        self.sessions[session_id] = session_data
        self.metrics["active_sessions"] += 1
        
        # Update agent last active
        self.agents[agent_id]["last_active"] = datetime.now().isoformat()
        
        return {
            "session_id": session_id,
            "status": "created",
            "message": "Session created successfully"
        }
    
    async def close_session(self, session_id: str) -> Dict[str, Any]:
        """Close an active session"""
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        session["status"] = "closed"
        session["closed_at"] = datetime.now().isoformat()
        
        self.metrics["active_sessions"] -= 1
        
        return {
            "session_id": session_id,
            "status": "closed",
            "message": "Session closed successfully",
            "execution_count": session["execution_count"]
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics"""
        return {
            **self.metrics,
            "total_tools": len(self.tools),
            "active_tools": sum(1 for t in self.tools.values() if t["status"] == "active"),
            "total_agents": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values() if a["status"] == "active"),
            "execution_history_size": len(self.execution_history)
        }
    
    def get_tool_metrics(self, tool_id: str) -> Dict[str, Any]:
        """Get metrics for a specific tool"""
        if tool_id not in self.tools:
            raise HTTPException(status_code=404, detail=f"Tool {tool_id} not found")
        
        tool = self.tools[tool_id]
        
        # Calculate success rate from execution history
        tool_executions = [e for e in self.execution_history if e["tool_id"] == tool_id]
        successful = sum(1 for e in tool_executions if e["status"] == "success")
        total = len(tool_executions)
        
        return {
            "tool_id": tool_id,
            "name": tool["name"],
            "usage_count": tool["usage_count"],
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": successful / total if total > 0 else 0,
            "last_used": tool_executions[-1]["timestamp"] if tool_executions else None
        }


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting MCP Server...")
    app.state.mcp_server = MCPServer()
    yield
    # Shutdown
    logger.info("Shutting down MCP Server...")


app = FastAPI(
    title="Enterprise MCP Server",
    description="Model Context Protocol Server for Financial Institutions",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()


# ============================================================================
# Authentication
# ============================================================================

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify JWT token and return agent ID"""
    token = credentials.credentials
    
    try:
        # In production, use proper JWT verification with secret key
        # This is a simplified example
        payload = jwt.decode(token, "secret_key", algorithms=["HS256"])
        agent_id = payload.get("agent_id")
        
        if not agent_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return agent_id
        
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MCP Server",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/tools/register")
async def register_tool(
    tool: ToolRegistration,
    agent_id: str = Depends(verify_token)
):
    """Register a new tool"""
    mcp_server = app.state.mcp_server
    result = await mcp_server.register_tool(tool, agent_id)
    return result


@app.get("/api/v1/tools/discover")
async def discover_tools(
    category: Optional[str] = None,
    permissions: Optional[List[str]] = None,
    agent_id: str = Depends(verify_token)
):
    """Discover available tools"""
    mcp_server = app.state.mcp_server
    tools = await mcp_server.discover_tools(agent_id, category, permissions)
    return {"tools": tools, "count": len(tools)}


@app.post("/api/v1/tools/execute")
async def execute_tool(
    request: ToolExecutionRequest,
    agent_id: str = Depends(verify_token)
):
    """Execute a tool"""
    mcp_server = app.state.mcp_server
    result = await mcp_server.execute_tool(request, agent_id)
    return result


@app.get("/api/v1/tools/{tool_id}/metrics")
async def get_tool_metrics(
    tool_id: str,
    agent_id: str = Depends(verify_token)
):
    """Get metrics for a specific tool"""
    mcp_server = app.state.mcp_server
    metrics = mcp_server.get_tool_metrics(tool_id)
    return metrics


@app.post("/api/v1/agents/register")
async def register_agent(agent: AgentRegistration):
    """Register a new agent (admin endpoint)"""
    # In production, this would require admin authentication
    mcp_server = app.state.mcp_server
    result = await mcp_server.register_agent(agent)
    return result


@app.post("/api/v1/sessions/create")
async def create_session(agent_id: str = Depends(verify_token)):
    """Create a new session"""
    mcp_server = app.state.mcp_server
    result = await mcp_server.create_session(agent_id)
    return result


@app.post("/api/v1/sessions/{session_id}/close")
async def close_session(
    session_id: str,
    agent_id: str = Depends(verify_token)
):
    """Close a session"""
    mcp_server = app.state.mcp_server
    
    # Verify session belongs to agent
    session = mcp_server.sessions.get(session_id)
    if not session or session["agent_id"] != agent_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    result = await mcp_server.close_session(session_id)
    return result


@app.get("/api/v1/metrics")
async def get_metrics(agent_id: str = Depends(verify_token)):
    """Get server metrics"""
    mcp_server = app.state.mcp_server
    metrics = mcp_server.get_metrics()
    return metrics


# ============================================================================
# WebSocket Support for Real-time Communication
# ============================================================================

from fastapi import WebSocket, WebSocketDisconnect
from typing import Set

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.agent_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, agent_id: str):
        """Accept new connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.agent_connections[agent_id] = websocket
        logger.info(f"Agent {agent_id} connected via WebSocket")
    
    def disconnect(self, websocket: WebSocket, agent_id: str):
        """Handle disconnection"""
        self.active_connections.discard(websocket)
        if agent_id in self.agent_connections:
            del self.agent_connections[agent_id]
        logger.info(f"Agent {agent_id} disconnected")
    
    async def send_to_agent(self, agent_id: str, message: Dict[str, Any]):
        """Send message to specific agent"""
        if agent_id in self.agent_connections:
            websocket = self.agent_connections[agent_id]
            await websocket.send_json(message)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected agents"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


@app.websocket("/ws/{agent_id}")
async def websocket_endpoint(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket, agent_id)
    
    try:
        while True:
            # Receive message from agent
            data = await websocket.receive_json()
            
            # Process message
            message_type = data.get("type")
            
            if message_type == "heartbeat":
                # Send heartbeat response
                await websocket.send_json({
                    "type": "heartbeat_ack",
                    "timestamp": datetime.now().isoformat()
                })
            
            elif message_type == "tool_execution":
                # Handle tool execution request
                mcp_server = app.state.mcp_server
                request = ToolExecutionRequest(**data["payload"])
                result = await mcp_server.execute_tool(request, agent_id)
                
                await websocket.send_json({
                    "type": "tool_result",
                    "result": result
                })
            
            elif message_type == "status_update":
                # Broadcast status update to other agents
                await manager.broadcast({
                    "type": "agent_status",
                    "agent_id": agent_id,
                    "status": data["status"],
                    "timestamp": datetime.now().isoformat()
                })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, agent_id)
    except Exception as e:
        logger.error(f"WebSocket error for agent {agent_id}: {str(e)}")
        manager.disconnect(websocket, agent_id)


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "mcp-server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )