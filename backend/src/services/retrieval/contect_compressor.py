"""
Context Compressor: Remove redundancy and limit token count
"""
from typing import List
from langchain.schema import Document


def compress_context(documents: List[Document], max_tokens: int = 2000) -> List[Document]:
    """
    Compress context by removing redundancy and limiting tokens
    
    Args:
        documents: List of documents to compress
        max_tokens: Maximum total tokens (rough estimate: 1 token ≈ 4 chars)
        
    Returns:
        Compressed document list
    """
    if not documents:
        return []
    
    compressed = []
    total_chars = 0
    max_chars = max_tokens * 4  # Rough conversion
    
    seen_content = set()
    
    for doc in documents:
        content = doc.page_content.strip()
        
        # Skip duplicates
        if content in seen_content:
            continue
        
        # Check token budget
        doc_chars = len(content)
        if total_chars + doc_chars > max_chars:
            # Truncate if this is the first doc
            if not compressed:
                remaining = max_chars - total_chars
                doc.page_content = content[:remaining] + "..."
                compressed.append(doc)
            break
        
        compressed.append(doc)
        seen_content.add(content)
        total_chars += doc_chars
    
    return compressed
