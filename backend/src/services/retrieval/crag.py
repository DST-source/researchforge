"""
CRAG: Corrective RAG with web fallback (Lance Martin Notebook 14)
"""
from typing import List
from langchain.schema import Document


class CRAG:
    """Corrective RAG: Check relevance → web fallback if needed"""
    
    def __init__(self, relevance_threshold: float = 0.5):
        self.threshold = relevance_threshold
    
    def check_relevance(self, docs: List[Document], query: str) -> bool:
        """Check if retrieved docs are relevant"""
        if not docs:
            return False
        
        # Simple heuristic: check if query keywords appear in top docs
        query_words = set(query.lower().split())
        top_doc_text = " ".join([d.page_content for d in docs[:3]]).lower()
        
        matches = sum(1 for word in query_words if word in top_doc_text)
        relevance = matches / len(query_words) if query_words else 0
        
        return relevance >= self.threshold
    
    def fallback_retrieve(self, query: str) -> List[Document]:
        """Web fallback (simplified - return empty for now)"""
        print(f"⚠️ CRAG: Low relevance detected, web fallback needed for: {query}")
        # TODO: Implement web search (Tavily/Serper API)
        return []
