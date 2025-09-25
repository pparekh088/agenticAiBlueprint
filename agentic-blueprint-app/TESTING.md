# Testing Guide for Agentic AI Blueprint Analyzer

## 🧪 Test Scenarios

### 1. Simple Use Case Test
**Input:** "I need a simple Q&A bot that answers customer questions"

**Expected Output:**
```json
{
  "reasoning_engine": false,
  "memory": false,
  "rag": false,
  "evaluation": false,
  "mcp_integration": false,
  "observability": true,
  "simple_direct_answer": true,
  "agents": []
}
```

### 2. Complex Use Case Test
**Input:** "Build a customer service chatbot that can access our CRM system, remember past conversations, and provide accurate responses based on our internal documentation"

**Expected Output:**
```json
{
  "reasoning_engine": true,
  "memory": true,
  "rag": true,
  "evaluation": true,
  "mcp_integration": true,
  "observability": true,
  "simple_direct_answer": false,
  "agents": ["agent_a", "agent_b"]
}
```

### 3. RAG-Focused Use Case
**Input:** "Create an AI system that searches through our technical documentation and provides detailed answers with citations"

**Expected Output:**
```json
{
  "reasoning_engine": false,
  "memory": false,
  "rag": true,
  "evaluation": true,
  "mcp_integration": false,
  "observability": true,
  "simple_direct_answer": false,
  "agents": ["agent_a"]
}
```

### 4. Multi-Agent Workflow
**Input:** "Develop an AI system for investment analysis that pulls data from multiple sources, performs risk assessment, generates reports, and monitors market conditions"

**Expected Output:**
```json
{
  "reasoning_engine": true,
  "memory": true,
  "rag": true,
  "evaluation": true,
  "mcp_integration": true,
  "observability": true,
  "simple_direct_answer": false,
  "agents": ["agent_a", "agent_b", "agent_c"]
}
```

## 🔍 Visual Testing

When testing the frontend visualization:

1. **Component Highlighting**: Verify that components light up with animation when selected
2. **Color Coding**: Check that each component type has its distinct color:
   - Reasoning Engine: Pink (#ec4899)
   - Memory (Redis): Red (#ef4444)
   - RAG: Cyan (#06b6d4)
   - Evaluation (RAGAS): Purple (#a855f7)
   - MCP Integration: Amber (#f59e0b)
   - Agents: Emerald (#10b981)
   - Observability: Orange (#f97316) - Always active

3. **Responsive Design**: Test on different screen sizes
4. **Error Handling**: Test with invalid inputs and network errors

## 🚀 Performance Testing

1. **Response Time**: API should respond within 2-5 seconds
2. **Concurrent Users**: Test with multiple simultaneous requests
3. **Large Inputs**: Test with maximum character limit (5000 chars)

## 🔐 Security Testing

1. **Input Validation**: Test with SQL injection attempts
2. **XSS Prevention**: Test with script tags in input
3. **CORS**: Verify only allowed origins can access API
4. **Authentication**: Verify DefaultAzureCredential works correctly

## 📊 API Testing with curl

### Health Check
```bash
curl http://localhost:8000/api/health
```

### Analyze Use Case
```bash
curl -X POST http://localhost:8000/api/analyze-usecase \
  -H "Content-Type: application/json" \
  -d '{"usecase": "Build a chatbot for customer service"}'
```

## 🐳 Docker Testing

### Build and Run
```bash
# Build image
docker build -t agentic-blueprint-app .

# Run container
docker run -p 8000:8000 \
  -e AZURE_OPENAI_ENDPOINT="your-endpoint" \
  -e AZURE_OPENAI_DEPLOYMENT="gpt-4" \
  agentic-blueprint-app
```

### Docker Compose
```bash
docker-compose up
```

## ✅ Validation Checklist

- [ ] Backend starts without errors
- [ ] Frontend builds successfully
- [ ] API endpoints respond correctly
- [ ] Azure OpenAI integration works
- [ ] Components highlight on analysis
- [ ] Observability layer is always visible
- [ ] Error messages are user-friendly
- [ ] Docker image builds and runs
- [ ] Health check passes
- [ ] CORS is properly configured