"""
Reranker: Cross-encoder reranking for better relevance
"""
from typing import List
from langchain.schema import Document


def rerank_documents(query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
    """
    Rerank documents using relevance scoring
    
    Args:
        query: Search query
        documents: List of documents to rerank
        top_k: Number of top documents to return
        
    Returns:
        Reranked documents
    """
    if not documents:
        return []
    
    try:
        # Simple relevance scoring (can be replaced with cross-encoder model)
        query_terms = set(query.lower().split())
        
        for doc in documents:
            doc_terms = set(doc.page_content.lower().split())
            
            # Calculate overlap score
            overlap = len(query_terms & doc_terms)
            coverage = overlap / len(query_terms) if query_terms else 0
            
            doc.metadata['rerank_score'] = coverage
        
        # Sort by rerank score
        reranked = sorted(documents, key=lambda x: x.metadata.get('rerank_score', 0), reverse=True)
        
        return reranked[:top_k]
    
    except Exception as e:
        print(f"⚠️ Reranking failed: {e}")
        return documents[:top_k]  # Fallback to original order
