"""
ChromaDB Uploader: Batch upload documents to ChromaDB
"""
from typing import List, Dict
import chromadb
from langchain.schema import Document
import uuid


class ChromaUploader:
    """Batch upload documents to ChromaDB collections"""
    
    def __init__(self, client: chromadb.HttpClient):
        self.client = client
    
    def upload_chunks(
        self,
        collection_name: str,
        documents: List[Document],
        embeddings: List[List[float]],
        batch_size: int = 100
    ) -> int:
        """
        Batch upload documents to ChromaDB
        
        Args:
            collection_name: Target collection
            documents: List of Document objects
            embeddings: Corresponding embeddings
            batch_size: Upload batch size
            
        Returns:
            Number of documents uploaded
        """
        if not documents:
            return 0
        
        # Get or create collection
        collection = self.client.get_or_create_collection(collection_name)
        
        # Batch upload
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_embs = embeddings[i:i+batch_size]
            
            collection.add(
                documents=[d.page_content for d in batch_docs],
                embeddings=batch_embs,
                metadatas=[d.metadata for d in batch_docs],
                ids=[str(uuid.uuid4()) for _ in batch_docs]
            )
        
        return len(documents)


# Test
if __name__ == "__main__":
    print("✅ ChromaUploader loaded (use via index_pipeline.py)")
