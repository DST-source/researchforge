"""
RAG Fusion: Reciprocal Rank Fusion (Lance Martin Notebook 8)
"""
from typing import List, Dict
from langchain.schema import Document
from collections import defaultdict


def reciprocal_rank_fusion(
    doc_lists: List[List[Document]], 
    k: int = 60
) -> List[Document]:
    """
    RRF: Fuse multiple ranked lists using reciprocal ranks
    
    Args:
        doc_lists: List of ranked document lists from different queries
        k: RRF constant (default 60)
    
    Returns:
        Fused and re-ranked document list
    """
    # Track scores by document ID
    doc_scores = defaultdict(float)
    doc_map = {}
    
    # Score each document based on its rank in each list
    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list, start=1):
            doc_id = doc.metadata.get("id", hash(doc.page_content))
            doc_scores[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = doc
    
    # Sort by fused score
    sorted_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return documents in fused order
    return [doc_map[doc_id] for doc_id, score in sorted_ids]
