"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
(Lance Martin Notebook 13)
Build hierarchical summaries via clustering + LLM summarization
"""
from typing import List, Dict
from langchain.schema import Document
from sklearn.cluster import KMeans
import numpy as np


def build_raptor_tree(
    chunks: List[Document],
    embedder,
    levels: int = 3,
    clusters_per_level: int = 5
) -> List[Dict]:
    """
    Build RAPTOR tree: cluster chunks → summarize → embed → repeat
    
    Args:
        chunks: Base-level chunks
        embedder: Embedder instance
        levels: Tree depth (0=leaf, 1=mid, 2=root)
        clusters_per_level: K for K-means
        
    Returns:
        List of dicts: {"summary": str, "level": int, "cluster_id": int}
    """
    all_nodes = []
    current_chunks = chunks
    
    for level in range(levels):
        print(f"   Building RAPTOR level {level}...")
        
        # Get embeddings for current level
        texts = [c.page_content if isinstance(c, Document) else c for c in current_chunks]
        embeddings = embedder.embed(texts)
        embeddings_array = np.array(embeddings)
        
        # K-means clustering
        n_clusters = min(clusters_per_level, len(current_chunks))
        if n_clusters < 2:
            break
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_array)
        
        # Summarize each cluster
        level_summaries = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_texts = [texts[i] for i in cluster_indices]
            
            # Combine cluster chunks
            combined = "\n\n".join(cluster_texts[:10])  # Max 10 chunks per cluster
            
            # Simple summarization (truncate for now - no LLM call)
            summary = summarize_cluster(combined, level)
            
            all_nodes.append({
                "summary": summary,
                "level": level,
                "cluster_id": cluster_id,
                "source_count": len(cluster_indices)
            })
            
            level_summaries.append(summary)
        
        # Next level uses summaries
        current_chunks = level_summaries
    
    return all_nodes


def summarize_cluster(text: str, level: int) -> str:
    """
    Summarize cluster of chunks
    
    For now: Simple truncation (no LLM needed)
    TODO: Add Ollama/Gemini summarization later
    """
    # Simple truncation - works without LLM
    max_length = 500
    if len(text) <= max_length:
        return text
    
    # Truncate at sentence boundary
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    if last_period > 300:  # Keep if reasonable
        return truncated[:last_period + 1]
    
    return truncated + "..."


# Test
if __name__ == "__main__":
    print("✅ RAPTOR module loaded (run via index_pipeline.py)")