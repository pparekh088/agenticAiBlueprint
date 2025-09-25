#!/usr/bin/env python3
"""
Test script to verify backend functionality
"""

import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def test_imports():
    """Test that all required imports work"""
    try:
        from backend.main import app, ComponentAnalysis, UseCaseRequest
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_models():
    """Test Pydantic models"""
    try:
        from backend.main import ComponentAnalysis, UseCaseRequest
        
        # Test UseCaseRequest
        request = UseCaseRequest(usecase="Test use case for validation")
        assert request.usecase == "Test use case for validation"
        
        # Test ComponentAnalysis
        analysis = ComponentAnalysis(
            reasoning_engine=True,
            memory=True,
            rag=False,
            evaluation=False,
            mcp_integration=True,
            observability=True,
            simple_direct_answer=False,
            agents=["agent_a", "agent_b"]
        )
        assert analysis.observability == True
        assert len(analysis.agents) == 2
        
        print("✅ Pydantic models working correctly")
        return True
    except Exception as e:
        print(f"❌ Model error: {e}")
        return False

def test_json_parsing():
    """Test JSON parsing for LLM response"""
    try:
        from backend.main import ComponentAnalysis
        
        sample_response = {
            "reasoning_engine": True,
            "memory": True,
            "rag": False,
            "evaluation": False,
            "mcp_integration": True,
            "observability": True,
            "simple_direct_answer": False,
            "agents": ["agent_a", "agent_c"]
        }
        
        analysis = ComponentAnalysis(**sample_response)
        assert analysis.reasoning_engine == True
        assert "agent_a" in analysis.agents
        
        print("✅ JSON parsing working correctly")
        return True
    except Exception as e:
        print(f"❌ JSON parsing error: {e}")
        return False

def main():
    print("🧪 Testing Backend Components\n")
    
    tests = [
        test_imports,
        test_models,
        test_json_parsing
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    if all(results):
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())