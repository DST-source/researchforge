"""
Multi-Query Retrieval: Generate multiple query perspectives
"""
from typing import List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from services.llm.client import chat


def generate_multi_queries(original_query: str, n: int = 3) -> List[str]:
    """
    Generate multiple query variants from original question
    
    Args:
        original_query: Original user question
        n: Number of query variants to generate
        
    Returns:
        List of query variants including original
    """
    prompt = f"""You are an AI research assistant. Generate {n} different ways to ask the following question.
Each variant should capture different aspects or perspectives of the original question.

Original Question: {original_query}

Generate {n} alternative questions (one per line):"""
    
    try:
        response = chat(
            [{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        
        # Parse response into list
        variants = [line.strip() for line in response.strip().split('\n') if line.strip()]
        
        # Add original query
        all_queries = [original_query] + variants[:n]
        
        return all_queries
    
    except Exception as e:
        print(f"⚠️ Multi-query generation failed: {e}")
        return [original_query]  # Fallback to original only
