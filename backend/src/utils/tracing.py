"""
LangSmith Tracing Utilities
Tracks all 4 phases with parent-child relationships
"""
import os
from functools import wraps
from typing import Dict, Any, List
from langsmith import traceable, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ResearchForge-Phase0")

langsmith_client = Client()


def trace_phase(phase_name: str, phase_num: int):
    """
    Decorator to trace a specific RAG phase
    
    Usage:
        @trace_phase("Query Construction", 1)
        def phase1_query_construction(question: str):
            ...
    """
    def decorator(func):
        @traceable(
            name=f"Phase {phase_num}: {phase_name}",
            run_type="chain",
            tags=[f"phase-{phase_num}", phase_name.lower().replace(" ", "-")],
            metadata={"phase": phase_num, "phase_name": phase_name}
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def trace_component(component_name: str, component_type: str = "retriever"):
    """
    Decorator to trace individual components within phases
    
    Types: retriever, llm, reranker, tool
    """
    def decorator(func):
        @traceable(
            name=component_name,
            run_type=component_type,
            tags=[component_type, component_name.lower().replace(" ", "-")]
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Convenience decorators
def trace_retrieval(name: str):
    """Trace a retrieval operation"""
    return trace_component(name, "retriever")


def trace_llm(name: str):
    """Trace an LLM call"""
    return trace_component(name, "llm")


def trace_tool(name: str):
    """Trace a tool/utility function"""
    return trace_component(name, "tool")
