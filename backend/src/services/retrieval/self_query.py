"""
Self-Query: Extract metadata filters from natural language queries
"""
from typing import Dict, Any, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from services.llm.client import chat


def extract_metadata_filters(query: str) -> Optional[Dict[str, Any]]:
    """
    Extract structured metadata filters from natural language query
    
    Args:
        query: Natural language query
        
    Returns:
        Dictionary of metadata filters or None
    """
    prompt = f"""Extract metadata filters from this query. Return ONLY a JSON object with filters.

Query: {query}

Possible filters: year, author, paper_name, section_type (introduction/methods/results/conclusion)

JSON (empty object if no filters):"""
    
    try:
        response = chat(
            [{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        
        # Try to parse JSON
        import json
        filters = json.loads(response.strip())
        
        return filters if filters else None
    
    except Exception as e:
        print(f"⚠️ Self-query filter extraction failed: {e}")
        return None
