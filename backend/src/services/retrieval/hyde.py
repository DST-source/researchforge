"""
HyDE: Hypothetical Document Embeddings (Lance Martin Notebook 6)
Generate fake answer → embed → better retrieval
"""
from typing import List
from ..llm.client import chat


def generate_hyde_document(query: str) -> str:
    """Generate hypothetical answer for better retrieval"""
    prompt = f"""Write a hypothetical research paper excerpt that would perfectly answer this question:

Question: {query}

Hypothetical excerpt (2-3 sentences):"""
    
    try:
        fake_doc = chat(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        return fake_doc.strip()
    except Exception as e:
        print(f"⚠️ HyDE generation failed: {e}")
        return query  # Fallback to original query
