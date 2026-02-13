"""
Embedder: Generate vector embeddings for text
Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim)
"""
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    """
    Vector embedding generator
    Model: all-MiniLM-L6-v2 (384 dimensions, fast, quality)
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            model_name: HuggingFace model name
        """
        print(f"🔧 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors (384-dim each)
        """
        if not texts:
            return []
        
        # Batch encoding for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 100
        )
        
        # Convert to list of lists
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed single query
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector (384-dim)
        """
        return self.model.encode(query).tolist()


# Test
if __name__ == "__main__":
    embedder = Embedder()
    test_texts = ["This is a test.", "Another test sentence."]
    embeddings = embedder.embed(test_texts)
    print(f"✅ Embedded {len(test_texts)} texts → {len(embeddings)} vectors")
    print(f"   Dimension: {len(embeddings[0])}")
