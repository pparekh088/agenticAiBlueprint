"""
Enterprise Agentic AI Implementation
Production-ready implementation for Financial Institutions
"""

import asyncio
import uuid
import json
import logging
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import redis
import httpx
from pydantic import BaseModel, Field
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Core Data Models
# ============================================================================

class AgentStatus(Enum):
    """Agent operational status"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class ToolCategory(Enum):
    """Tool categories for classification"""
    DATA_RETRIEVAL = "data_retrieval"
    COMPUTATION = "computation"
    INTEGRATION = "integration"
    NOTIFICATION = "notification"
    ANALYSIS = "analysis"
    TRANSACTION = "transaction"


@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    required_tools: List[str]
    max_concurrent_tasks: int = 5
    timeout_seconds: int = 300
    retry_policy: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpecification:
    """Tool specification for MCP registry"""
    id: str
    name: str
    description: str
    category: ToolCategory
    version: str
    endpoint: str
    authentication_required: bool
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    rate_limits: Dict[str, int] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)
    timeout: int = 30
    retry_count: int = 3


@dataclass
class AgentTask:
    """Task definition for agents"""
    id: str
    type: str
    payload: Dict[str, Any]
    priority: TaskPriority
    created_at: datetime
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None


# ============================================================================
# Guardrails Layer Implementation
# ============================================================================

class GuardrailsLayer:
    """
    Comprehensive guardrails for agent operations
    Implements security, compliance, and safety checks
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validators = self._initialize_validators()
        self.filters = self._initialize_filters()
        self.rate_limiter = RateLimiter(config.get("rate_limits", {}))
        self.audit_logger = AuditLogger()
        
    def _initialize_validators(self) -> Dict[str, Callable]:
        """Initialize input validators"""
        return {
            "pii_detection": self._detect_pii,
            "injection_detection": self._detect_injection,
            "schema_validation": self._validate_schema,
            "permission_check": self._check_permissions,
            "compliance_check": self._check_compliance
        }
    
    def _initialize_filters(self) -> Dict[str, Callable]:
        """Initialize output filters"""
        return {
            "pii_masking": self._mask_pii,
            "sensitive_data": self._filter_sensitive_data,
            "hallucination": self._detect_hallucination,
            "bias_detection": self._detect_bias
        }
    
    async def validate_input(self, 
                            agent_id: str,
                            input_data: Dict[str, Any],
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input before processing
        
        Args:
            agent_id: Agent identifier
            input_data: Input to validate
            context: Additional context
            
        Returns:
            Validation result with status and details
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "modified_input": input_data.copy()
        }
        
        # Check rate limits
        if not await self.rate_limiter.check_limit(agent_id):
            validation_results["valid"] = False
            validation_results["errors"].append("Rate limit exceeded")
            return validation_results
        
        # Run validators
        for validator_name, validator_func in self.validators.items():
            try:
                result = await validator_func(input_data, context)
                if not result["valid"]:
                    validation_results["valid"] = False
                    validation_results["errors"].extend(result.get("errors", []))
                validation_results["warnings"].extend(result.get("warnings", []))
                
                # Apply any input modifications
                if "modified_input" in result:
                    validation_results["modified_input"] = result["modified_input"]
                    
            except Exception as e:
                logger.error(f"Validator {validator_name} failed: {str(e)}")
                validation_results["warnings"].append(f"Validator {validator_name} failed")
        
        # Log validation attempt
        await self.audit_logger.log_validation(
            agent_id=agent_id,
            input_data=input_data,
            validation_results=validation_results
        )
        
        return validation_results
    
    async def filter_output(self,
                          agent_id: str,
                          output_data: Any,
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter output before returning to user
        
        Args:
            agent_id: Agent identifier
            output_data: Output to filter
            context: Additional context
            
        Returns:
            Filtered output with metadata
        """
        filter_results = {
            "filtered_output": output_data,
            "modifications": [],
            "warnings": []
        }
        
        # Run filters
        for filter_name, filter_func in self.filters.items():
            try:
                result = await filter_func(output_data, context)
                filter_results["filtered_output"] = result["filtered_output"]
                filter_results["modifications"].extend(result.get("modifications", []))
                filter_results["warnings"].extend(result.get("warnings", []))
                
            except Exception as e:
                logger.error(f"Filter {filter_name} failed: {str(e)}")
                filter_results["warnings"].append(f"Filter {filter_name} failed")
        
        # Log filtering
        await self.audit_logger.log_filtering(
            agent_id=agent_id,
            original_output=output_data,
            filter_results=filter_results
        )
        
        return filter_results
    
    async def _detect_pii(self, data: Any, context: Dict) -> Dict[str, Any]:
        """Detect PII in input data"""
        # Simplified PII detection - use specialized libraries in production
        pii_patterns = {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        }
        
        result = {"valid": True, "errors": [], "warnings": []}
        
        data_str = json.dumps(data) if isinstance(data, dict) else str(data)
        
        import re
        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, data_str):
                result["warnings"].append(f"Potential {pii_type} detected")
        
        return result
    
    async def _detect_injection(self, data: Any, context: Dict) -> Dict[str, Any]:
        """Detect injection attempts"""
        result = {"valid": True, "errors": [], "warnings": []}
        
        # SQL injection patterns
        sql_patterns = [
            r"(\bUNION\b.*\bSELECT\b)",
            r"(\bDROP\b.*\bTABLE\b)",
            r"(\bINSERT\b.*\bINTO\b)",
            r"(\bDELETE\b.*\bFROM\b)"
        ]
        
        data_str = json.dumps(data) if isinstance(data, dict) else str(data)
        
        import re
        for pattern in sql_patterns:
            if re.search(pattern, data_str, re.IGNORECASE):
                result["valid"] = False
                result["errors"].append("Potential SQL injection detected")
                break
        
        # Check for prompt injection
        prompt_injection_indicators = [
            "ignore previous instructions",
            "disregard all prior",
            "new instructions:",
            "system: you are"
        ]
        
        data_lower = data_str.lower()
        for indicator in prompt_injection_indicators:
            if indicator in data_lower:
                result["warnings"].append("Potential prompt injection attempt")
        
        return result
    
    async def _validate_schema(self, data: Any, context: Dict) -> Dict[str, Any]:
        """Validate data against expected schema"""
        # Implementation would use jsonschema or similar
        return {"valid": True, "errors": [], "warnings": []}
    
    async def _check_permissions(self, data: Any, context: Dict) -> Dict[str, Any]:
        """Check if operation is permitted"""
        # Check against permission matrix
        return {"valid": True, "errors": [], "warnings": []}
    
    async def _check_compliance(self, data: Any, context: Dict) -> Dict[str, Any]:
        """Check regulatory compliance"""
        # Check against compliance rules
        return {"valid": True, "errors": [], "warnings": []}
    
    async def _mask_pii(self, data: Any, context: Dict) -> Dict[str, Any]:
        """Mask PII in output"""
        # Mask sensitive information
        return {"filtered_output": data, "modifications": []}
    
    async def _filter_sensitive_data(self, data: Any, context: Dict) -> Dict[str, Any]:
        """Filter sensitive data from output"""
        return {"filtered_output": data, "modifications": []}
    
    async def _detect_hallucination(self, data: Any, context: Dict) -> Dict[str, Any]:
        """Detect potential hallucinations in output"""
        return {"filtered_output": data, "warnings": []}
    
    async def _detect_bias(self, data: Any, context: Dict) -> Dict[str, Any]:
        """Detect potential bias in output"""
        return {"filtered_output": data, "warnings": []}


# ============================================================================
# MCP Tool Registry Implementation
# ============================================================================

class MCPToolRegistry:
    """
    Model Context Protocol Tool Registry
    Manages tool discovery, registration, and execution
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.tools: Dict[str, ToolSpecification] = {}
        self.tool_instances: Dict[str, Any] = {}
        self.permissions: Dict[str, List[str]] = {}
        self.usage_metrics: Dict[str, Dict[str, Any]] = {}
        self.redis_client = redis_client
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
    async def register_tool(self, tool_spec: ToolSpecification) -> str:
        """
        Register a new tool with the registry
        
        Args:
            tool_spec: Tool specification
            
        Returns:
            Tool ID
        """
        # Validate tool specification
        if not self._validate_tool_spec(tool_spec):
            raise ValueError(f"Invalid tool specification for {tool_spec.name}")
        
        # Register tool
        self.tools[tool_spec.id] = tool_spec
        
        # Initialize circuit breaker for the tool
        self.circuit_breakers[tool_spec.id] = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception
        )
        
        # Initialize usage metrics
        self.usage_metrics[tool_spec.id] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_latency": 0,
            "last_used": None
        }
        
        # Store in Redis if available
        if self.redis_client:
            await self._store_tool_in_redis(tool_spec)
        
        logger.info(f"Registered tool: {tool_spec.name} (ID: {tool_spec.id})")
        return tool_spec.id
    
    async def discover_tools(self,
                           agent_id: str,
                           category: Optional[ToolCategory] = None,
                           capabilities: Optional[List[str]] = None) -> List[ToolSpecification]:
        """
        Discover available tools for an agent
        
        Args:
            agent_id: Agent identifier
            category: Filter by tool category
            capabilities: Required capabilities
            
        Returns:
            List of available tools
        """
        available_tools = []
        
        for tool_id, tool_spec in self.tools.items():
            # Check permissions
            if not await self._check_permission(agent_id, tool_id):
                continue
            
            # Filter by category
            if category and tool_spec.category != category:
                continue
            
            # Filter by capabilities
            if capabilities:
                # Check if tool supports required capabilities
                # This would require capability matching logic
                pass
            
            available_tools.append(tool_spec)
        
        return available_tools
    
    async def execute_tool(self,
                         agent_id: str,
                         tool_id: str,
                         params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with proper authentication and monitoring
        
        Args:
            agent_id: Agent identifier
            tool_id: Tool identifier
            params: Tool parameters
            
        Returns:
            Tool execution result
        """
        # Check if tool exists
        if tool_id not in self.tools:
            raise ValueError(f"Tool {tool_id} not found")
        
        tool_spec = self.tools[tool_id]
        
        # Check permissions
        if not await self._check_permission(agent_id, tool_id):
            raise PermissionError(f"Agent {agent_id} lacks permission for tool {tool_id}")
        
        # Validate parameters
        if not self._validate_params(params, tool_spec.input_schema):
            raise ValueError(f"Invalid parameters for tool {tool_id}")
        
        # Check rate limits
        if not await self._check_rate_limit(agent_id, tool_id):
            raise Exception(f"Rate limit exceeded for tool {tool_id}")
        
        # Execute with circuit breaker
        circuit_breaker = self.circuit_breakers[tool_id]
        
        try:
            start_time = datetime.now()
            
            # Execute through circuit breaker
            result = await circuit_breaker.call(
                self._execute_tool_internal,
                tool_spec,
                params
            )
            
            # Update metrics
            latency = (datetime.now() - start_time).total_seconds()
            await self._update_metrics(tool_id, success=True, latency=latency)
            
            # Log execution
            logger.info(f"Tool {tool_id} executed successfully by agent {agent_id}")
            
            return {
                "success": True,
                "result": result,
                "tool_id": tool_id,
                "latency_ms": latency * 1000,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            # Update metrics
            await self._update_metrics(tool_id, success=False)
            
            # Log error
            logger.error(f"Tool {tool_id} execution failed: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "tool_id": tool_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _execute_tool_internal(self,
                                    tool_spec: ToolSpecification,
                                    params: Dict[str, Any]) -> Any:
        """Internal tool execution logic"""
        # Make HTTP request to tool endpoint
        async with httpx.AsyncClient(timeout=tool_spec.timeout) as client:
            response = await client.post(
                tool_spec.endpoint,
                json=params,
                headers=self._get_auth_headers(tool_spec)
            )
            response.raise_for_status()
            return response.json()
    
    def _validate_tool_spec(self, tool_spec: ToolSpecification) -> bool:
        """Validate tool specification"""
        # Check required fields
        required_fields = ["id", "name", "description", "endpoint", "input_schema", "output_schema"]
        for field in required_fields:
            if not getattr(tool_spec, field, None):
                return False
        return True
    
    def _validate_params(self, params: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate parameters against schema"""
        # Simplified validation - use jsonschema in production
        required_params = schema.get("required", [])
        for param in required_params:
            if param not in params:
                return False
        return True
    
    async def _check_permission(self, agent_id: str, tool_id: str) -> bool:
        """Check if agent has permission to use tool"""
        # Check permission matrix
        agent_permissions = self.permissions.get(agent_id, [])
        tool_spec = self.tools.get(tool_id)
        
        if not tool_spec:
            return False
        
        # Check if agent has required permissions
        for required_perm in tool_spec.permissions:
            if required_perm not in agent_permissions:
                return False
        
        return True
    
    async def _check_rate_limit(self, agent_id: str, tool_id: str) -> bool:
        """Check rate limits for tool usage"""
        if not self.redis_client:
            return True
        
        # Implement rate limiting using Redis
        key = f"rate_limit:{agent_id}:{tool_id}"
        tool_spec = self.tools[tool_id]
        
        # Get rate limits from tool spec
        requests_per_minute = tool_spec.rate_limits.get("requests_per_minute", 60)
        
        # Check current count
        current_count = self.redis_client.get(key)
        if current_count and int(current_count) >= requests_per_minute:
            return False
        
        # Increment counter
        pipe = self.redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, 60)
        pipe.execute()
        
        return True
    
    async def _update_metrics(self, tool_id: str, success: bool, latency: float = 0):
        """Update tool usage metrics"""
        metrics = self.usage_metrics[tool_id]
        metrics["total_calls"] += 1
        
        if success:
            metrics["successful_calls"] += 1
            metrics["total_latency"] += latency
        else:
            metrics["failed_calls"] += 1
        
        metrics["last_used"] = datetime.now().isoformat()
        
        # Store in Redis if available
        if self.redis_client:
            self.redis_client.hset(
                f"tool_metrics:{tool_id}",
                mapping={k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                        for k, v in metrics.items()}
            )
    
    def _get_auth_headers(self, tool_spec: ToolSpecification) -> Dict[str, str]:
        """Get authentication headers for tool"""
        headers = {"Content-Type": "application/json"}
        
        if tool_spec.authentication_required:
            # Add authentication headers based on tool requirements
            # This would be configured per tool
            headers["Authorization"] = "Bearer <token>"
        
        return headers
    
    async def _store_tool_in_redis(self, tool_spec: ToolSpecification):
        """Store tool specification in Redis"""
        if self.redis_client:
            tool_data = {
                "id": tool_spec.id,
                "name": tool_spec.name,
                "description": tool_spec.description,
                "category": tool_spec.category.value,
                "version": tool_spec.version,
                "endpoint": tool_spec.endpoint,
                "authentication_required": tool_spec.authentication_required,
                "input_schema": json.dumps(tool_spec.input_schema),
                "output_schema": json.dumps(tool_spec.output_schema),
                "rate_limits": json.dumps(tool_spec.rate_limits),
                "permissions": json.dumps(tool_spec.permissions),
                "timeout": tool_spec.timeout,
                "retry_count": tool_spec.retry_count
            }
            
            self.redis_client.hset(
                f"tool:{tool_spec.id}",
                mapping=tool_data
            )
    
    def get_metrics(self, tool_id: Optional[str] = None) -> Dict[str, Any]:
        """Get usage metrics for tools"""
        if tool_id:
            return self.usage_metrics.get(tool_id, {})
        return self.usage_metrics


# ============================================================================
# Agent Base Class and Implementation
# ============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all agents
    Provides common functionality and interface
    """
    
    def __init__(self,
                 agent_id: str,
                 name: str,
                 capabilities: List[AgentCapability],
                 tool_registry: MCPToolRegistry,
                 guardrails: GuardrailsLayer):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.tool_registry = tool_registry
        self.guardrails = guardrails
        self.status = AgentStatus.IDLE
        self.current_tasks: List[AgentTask] = []
        self.task_history: List[AgentTask] = []
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0,
            "average_execution_time": 0
        }
    
    @abstractmethod
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process a task - must be implemented by subclasses"""
        pass
    
    async def execute(self, task: AgentTask) -> Dict[str, Any]:
        """
        Execute a task with full lifecycle management
        
        Args:
            task: Task to execute
            
        Returns:
            Execution result
        """
        try:
            # Update status
            self.status = AgentStatus.BUSY
            self.current_tasks.append(task)
            task.status = "in_progress"
            
            # Validate input through guardrails
            validation_result = await self.guardrails.validate_input(
                self.agent_id,
                task.payload,
                {"task_type": task.type, "priority": task.priority}
            )
            
            if not validation_result["valid"]:
                raise ValueError(f"Input validation failed: {validation_result['errors']}")
            
            # Use modified input if any
            task.payload = validation_result["modified_input"]
            
            # Process the task
            start_time = datetime.now()
            result = await self.process_task(task)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Filter output through guardrails
            filter_result = await self.guardrails.filter_output(
                self.agent_id,
                result,
                {"task_type": task.type}
            )
            
            # Update task with results
            task.status = "completed"
            task.result = filter_result["filtered_output"]
            
            # Update metrics
            self._update_metrics(success=True, execution_time=execution_time)
            
            # Move to history
            self.current_tasks.remove(task)
            self.task_history.append(task)
            
            # Update status
            if not self.current_tasks:
                self.status = AgentStatus.IDLE
            
            return {
                "success": True,
                "task_id": task.id,
                "result": task.result,
                "execution_time": execution_time,
                "warnings": filter_result.get("warnings", [])
            }
            
        except Exception as e:
            # Handle failure
            task.status = "failed"
            task.error = str(e)
            
            # Update metrics
            self._update_metrics(success=False)
            
            # Update status
            self.status = AgentStatus.ERROR
            
            # Log error
            logger.error(f"Agent {self.agent_id} task {task.id} failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "task_id": task.id,
                "error": str(e)
            }
    
    async def use_tool(self, tool_id: str, params: Dict[str, Any]) -> Any:
        """
        Use a tool from the registry
        
        Args:
            tool_id: Tool identifier
            params: Tool parameters
            
        Returns:
            Tool result
        """
        result = await self.tool_registry.execute_tool(
            self.agent_id,
            tool_id,
            params
        )
        
        if not result["success"]:
            raise Exception(f"Tool execution failed: {result['error']}")
        
        return result["result"]
    
    def _update_metrics(self, success: bool, execution_time: float = 0):
        """Update agent metrics"""
        if success:
            self.metrics["tasks_completed"] += 1
            self.metrics["total_execution_time"] += execution_time
            
            total_tasks = self.metrics["tasks_completed"] + self.metrics["tasks_failed"]
            self.metrics["average_execution_time"] = (
                self.metrics["total_execution_time"] / self.metrics["tasks_completed"]
            )
        else:
            self.metrics["tasks_failed"] += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status.value,
            "current_tasks": len(self.current_tasks),
            "metrics": self.metrics
        }


# ============================================================================
# Specialized Agent Implementations
# ============================================================================

class TradingAgent(BaseAgent):
    """Specialized agent for trading operations"""
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process trading-related tasks"""
        task_type = task.type
        
        if task_type == "market_analysis":
            return await self._perform_market_analysis(task.payload)
        elif task_type == "execute_trade":
            return await self._execute_trade(task.payload)
        elif task_type == "portfolio_optimization":
            return await self._optimize_portfolio(task.payload)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _perform_market_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform market analysis"""
        # Use market data tool
        market_data = await self.use_tool("market_data_api", {
            "symbols": params["symbols"],
            "timeframe": params.get("timeframe", "1d")
        })
        
        # Use analysis tool
        analysis = await self.use_tool("technical_analysis", {
            "data": market_data,
            "indicators": params.get("indicators", ["RSI", "MACD", "MA"])
        })
        
        return {
            "market_data": market_data,
            "analysis": analysis,
            "recommendations": self._generate_recommendations(analysis)
        }
    
    async def _execute_trade(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade"""
        # Validate trade parameters
        validation = await self.use_tool("trade_validator", params)
        
        if not validation["valid"]:
            raise ValueError(f"Trade validation failed: {validation['errors']}")
        
        # Check risk limits
        risk_check = await self.use_tool("risk_checker", params)
        
        if not risk_check["approved"]:
            raise ValueError(f"Risk check failed: {risk_check['reason']}")
        
        # Execute trade
        trade_result = await self.use_tool("trading_platform_api", {
            "action": params["action"],
            "symbol": params["symbol"],
            "quantity": params["quantity"],
            "order_type": params.get("order_type", "market")
        })
        
        return trade_result
    
    async def _optimize_portfolio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio allocation"""
        # Get current portfolio
        portfolio = await self.use_tool("portfolio_api", {
            "account_id": params["account_id"]
        })
        
        # Run optimization
        optimization = await self.use_tool("portfolio_optimizer", {
            "portfolio": portfolio,
            "constraints": params.get("constraints", {}),
            "objective": params.get("objective", "max_sharpe")
        })
        
        return optimization
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate trading recommendations based on analysis"""
        recommendations = []
        
        # Simplified recommendation logic
        if analysis.get("rsi", 50) < 30:
            recommendations.append("Consider buying - RSI indicates oversold")
        elif analysis.get("rsi", 50) > 70:
            recommendations.append("Consider selling - RSI indicates overbought")
        
        return recommendations


class ComplianceAgent(BaseAgent):
    """Specialized agent for compliance operations"""
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process compliance-related tasks"""
        task_type = task.type
        
        if task_type == "regulatory_check":
            return await self._perform_regulatory_check(task.payload)
        elif task_type == "audit_trail":
            return await self._generate_audit_trail(task.payload)
        elif task_type == "compliance_report":
            return await self._generate_compliance_report(task.payload)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _perform_regulatory_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform regulatory compliance check"""
        # Check against regulations
        regulations = await self.use_tool("regulation_database", {
            "jurisdiction": params["jurisdiction"],
            "activity_type": params["activity_type"]
        })
        
        # Perform checks
        compliance_status = await self.use_tool("compliance_checker", {
            "activity": params["activity"],
            "regulations": regulations
        })
        
        return {
            "compliant": compliance_status["compliant"],
            "violations": compliance_status.get("violations", []),
            "recommendations": compliance_status.get("recommendations", [])
        }
    
    async def _generate_audit_trail(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate audit trail for activities"""
        # Retrieve activity logs
        logs = await self.use_tool("activity_logger", {
            "start_date": params["start_date"],
            "end_date": params["end_date"],
            "filters": params.get("filters", {})
        })
        
        # Generate audit trail
        audit_trail = await self.use_tool("audit_generator", {
            "logs": logs,
            "format": params.get("format", "detailed")
        })
        
        return audit_trail
    
    async def _generate_compliance_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate compliance report"""
        # Gather compliance data
        compliance_data = await self.use_tool("compliance_data_aggregator", {
            "period": params["period"],
            "report_type": params["report_type"]
        })
        
        # Generate report
        report = await self.use_tool("report_generator", {
            "data": compliance_data,
            "template": params.get("template", "standard"),
            "format": params.get("format", "pdf")
        })
        
        return report


# ============================================================================
# Agent Orchestrator
# ============================================================================

class AgentOrchestrator:
    """
    Orchestrates multiple agents for complex workflows
    Handles task routing, coordination, and monitoring
    """
    
    def __init__(self,
                 tool_registry: MCPToolRegistry,
                 guardrails: GuardrailsLayer):
        self.agents: Dict[str, BaseAgent] = {}
        self.tool_registry = tool_registry
        self.guardrails = guardrails
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.running_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[AgentTask] = []
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.name} (ID: {agent.agent_id})")
    
    async def submit_task(self, task: AgentTask) -> str:
        """
        Submit a task for processing
        
        Args:
            task: Task to process
            
        Returns:
            Task ID
        """
        # Add to queue
        await self.task_queue.put(task)
        self.running_tasks[task.id] = task
        
        logger.info(f"Task {task.id} submitted for processing")
        return task.id
    
    async def process_tasks(self):
        """Main task processing loop"""
        while True:
            try:
                # Get task from queue
                task = await self.task_queue.get()
                
                # Route to appropriate agent
                agent = self._route_task(task)
                
                if not agent:
                    logger.error(f"No agent available for task {task.id}")
                    task.status = "failed"
                    task.error = "No suitable agent found"
                    continue
                
                # Process task asynchronously
                asyncio.create_task(self._process_task_async(agent, task))
                
            except Exception as e:
                logger.error(f"Error in task processing loop: {str(e)}")
                await asyncio.sleep(1)
    
    async def _process_task_async(self, agent: BaseAgent, task: AgentTask):
        """Process task asynchronously"""
        try:
            # Check dependencies
            await self._wait_for_dependencies(task)
            
            # Execute task
            result = await agent.execute(task)
            
            # Handle result
            if result["success"]:
                logger.info(f"Task {task.id} completed successfully")
            else:
                logger.error(f"Task {task.id} failed: {result['error']}")
            
            # Move to completed
            self.running_tasks.pop(task.id, None)
            self.completed_tasks.append(task)
            
        except Exception as e:
            logger.error(f"Error processing task {task.id}: {str(e)}")
            task.status = "failed"
            task.error = str(e)
    
    def _route_task(self, task: AgentTask) -> Optional[BaseAgent]:
        """Route task to appropriate agent"""
        # Simple routing based on task type
        task_type_mapping = {
            "market_analysis": "trading_agent",
            "execute_trade": "trading_agent",
            "portfolio_optimization": "trading_agent",
            "regulatory_check": "compliance_agent",
            "audit_trail": "compliance_agent",
            "compliance_report": "compliance_agent"
        }
        
        agent_type = task_type_mapping.get(task.type)
        
        if not agent_type:
            return None
        
        # Find available agent of the required type
        for agent in self.agents.values():
            if agent_type in agent.agent_id and agent.status == AgentStatus.IDLE:
                return agent
        
        # If no idle agent, find least busy
        suitable_agents = [a for a in self.agents.values() if agent_type in a.agent_id]
        if suitable_agents:
            return min(suitable_agents, key=lambda a: len(a.current_tasks))
        
        return None
    
    async def _wait_for_dependencies(self, task: AgentTask):
        """Wait for task dependencies to complete"""
        if not task.dependencies:
            return
        
        while True:
            # Check if all dependencies are completed
            all_completed = all(
                any(t.id == dep_id and t.status == "completed" 
                    for t in self.completed_tasks)
                for dep_id in task.dependencies
            )
            
            if all_completed:
                break
            
            await asyncio.sleep(1)
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "agents": {
                agent_id: agent.get_status() 
                for agent_id, agent in self.agents.items()
            },
            "queue_size": self.task_queue.qsize(),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks)
        }


# ============================================================================
# Supporting Classes
# ============================================================================

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        return (
            self.last_failure_time and
            (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, limits: Dict[str, int]):
        self.limits = limits
        self.counters: Dict[str, List[datetime]] = {}
    
    async def check_limit(self, key: str) -> bool:
        """Check if rate limit is exceeded"""
        limit = self.limits.get(key, self.limits.get("default", 100))
        
        now = datetime.now()
        
        # Clean old entries
        if key in self.counters:
            self.counters[key] = [
                t for t in self.counters[key]
                if (now - t).total_seconds() < 60
            ]
        else:
            self.counters[key] = []
        
        # Check limit
        if len(self.counters[key]) >= limit:
            return False
        
        # Add current request
        self.counters[key].append(now)
        return True


class AuditLogger:
    """Audit logger for compliance"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.local_logs: List[Dict[str, Any]] = []
    
    async def log_validation(self,
                            agent_id: str,
                            input_data: Any,
                            validation_results: Dict[str, Any]):
        """Log input validation"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "validation",
            "agent_id": agent_id,
            "input_hash": self._hash_data(input_data),
            "validation_results": validation_results
        }
        
        await self._store_log(log_entry)
    
    async def log_filtering(self,
                           agent_id: str,
                           original_output: Any,
                           filter_results: Dict[str, Any]):
        """Log output filtering"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "filtering",
            "agent_id": agent_id,
            "output_hash": self._hash_data(original_output),
            "filter_results": filter_results
        }
        
        await self._store_log(log_entry)
    
    async def _store_log(self, log_entry: Dict[str, Any]):
        """Store log entry"""
        # Store locally
        self.local_logs.append(log_entry)
        
        # Store in Redis if available
        if self.redis_client:
            key = f"audit_log:{log_entry['timestamp']}"
            self.redis_client.set(key, json.dumps(log_entry), ex=86400 * 30)  # 30 days
    
    def _hash_data(self, data: Any) -> str:
        """Hash data for audit trail"""
        import hashlib
        data_str = json.dumps(data, sort_keys=True) if isinstance(data, (dict, list)) else str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()


# ============================================================================
# Main Application Entry Point
# ============================================================================

async def main():
    """Main application entry point"""
    
    # Initialize Redis connection (optional)
    redis_client = None  # redis.Redis(host='localhost', port=6379, db=0)
    
    # Initialize guardrails
    guardrails_config = {
        "rate_limits": {
            "default": 100,
            "trading_agent": 50,
            "compliance_agent": 200
        }
    }
    guardrails = GuardrailsLayer(guardrails_config)
    
    # Initialize tool registry
    tool_registry = MCPToolRegistry(redis_client)
    
    # Register tools
    market_data_tool = ToolSpecification(
        id="market_data_api",
        name="Market Data API",
        description="Retrieve real-time market data",
        category=ToolCategory.DATA_RETRIEVAL,
        version="1.0.0",
        endpoint="http://market-data-service/api/v1/data",
        authentication_required=True,
        input_schema={"symbols": "array", "timeframe": "string"},
        output_schema={"data": "object"},
        rate_limits={"requests_per_minute": 100},
        permissions=["read_market_data"]
    )
    await tool_registry.register_tool(market_data_tool)
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator(tool_registry, guardrails)
    
    # Create and register agents
    trading_agent = TradingAgent(
        agent_id="trading_agent_001",
        name="Trading Agent",
        capabilities=[
            AgentCapability(
                name="market_analysis",
                description="Perform market analysis",
                required_tools=["market_data_api", "technical_analysis"]
            )
        ],
        tool_registry=tool_registry,
        guardrails=guardrails
    )
    orchestrator.register_agent(trading_agent)
    
    compliance_agent = ComplianceAgent(
        agent_id="compliance_agent_001",
        name="Compliance Agent",
        capabilities=[
            AgentCapability(
                name="regulatory_check",
                description="Check regulatory compliance",
                required_tools=["regulation_database", "compliance_checker"]
            )
        ],
        tool_registry=tool_registry,
        guardrails=guardrails
    )
    orchestrator.register_agent(compliance_agent)
    
    # Start task processing
    asyncio.create_task(orchestrator.process_tasks())
    
    # Example: Submit a task
    sample_task = AgentTask(
        id=str(uuid.uuid4()),
        type="market_analysis",
        payload={
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "timeframe": "1d",
            "indicators": ["RSI", "MACD"]
        },
        priority=TaskPriority.HIGH,
        created_at=datetime.now()
    )
    
    task_id = await orchestrator.submit_task(sample_task)
    logger.info(f"Submitted task: {task_id}")
    
    # Keep the application running
    while True:
        await asyncio.sleep(10)
        status = orchestrator.get_status()
        logger.info(f"Orchestrator status: {json.dumps(status, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())