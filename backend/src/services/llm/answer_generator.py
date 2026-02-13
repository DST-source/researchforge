"""
Answer Generation with Citations
"""
from typing import List, Dict
from langchain.schema import Document
from .client import chat
from ..retrieval.citation_tracker import CitationTracker


class AnswerGenerator:
    """Generate grounded answers with citations"""
    
    def __init__(self):
        self.tracker = CitationTracker()
    
    def generate(self, question: str, context_docs: List[Document]) -> Dict:
        """Generate answer with citations"""
        
        # Format context with citation markers
        context = self.tracker.format_context_with_citations(context_docs)
        
        # Build prompt
        prompt = f"""You are a research assistant. Answer the question using ONLY the provided context.

IMPORTANT: Cite your sources using [1], [2], etc. matching the numbered excerpts below.

Context:
{context}

Question: {question}

Answer (with citations):"""
        
        # Generate answer
        try:
            answer = chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            # Extract used citations
            used_citations = self.tracker.extract_citations(answer)
            
            return {
                "answer": answer,
                "citations": used_citations,
                "sources": [context_docs[i-1] for i in used_citations if 0 < i <= len(context_docs)]
            }
        
        except Exception as e:
            return {
                "answer": f"Error generating answer: {e}",
                "citations": [],
                "sources": []
            }
