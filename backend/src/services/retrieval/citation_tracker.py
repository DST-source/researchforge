"""
Citation Tracking: Map answers back to source documents
"""
from typing import List, Dict
from langchain.schema import Document


class CitationTracker:
    """Track which parts of answer came from which sources"""
    
    def format_context_with_citations(self, docs: List[Document]) -> str:
        """Format documents with citation markers"""
        context = ""
        for i, doc in enumerate(docs, start=1):
            paper = doc.metadata.get("paper_name", "Unknown")
            page = doc.metadata.get("page", "?")
            
            context += f"[{i}] {doc.page_content}\n"
            context += f"   (Source: {paper}, page {page})\n\n"
        
        return context
    
    def extract_citations(self, answer: str) -> List[int]:
        """Extract citation numbers from answer text"""
        import re
        citations = re.findall(r'\[(\d+)\]', answer)
        return [int(c) for c in citations]
