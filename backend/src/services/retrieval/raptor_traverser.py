"""
RAPTOR Tree Traversal: Query hierarchical summary tree
"""
from typing import List
import chromadb
from langchain.schema import Document


def query_raptor_tree(chroma_client: chromadb.HttpClient, query: str, k: int = 3) -> List[Document]:
    """
    Query RAPTOR tree for high-level summaries
    
    Args:
        chroma_client: ChromaDB client
        query: Search query
        k: Number of summaries to return
        
    Returns:
        List of summary documents
    """
    try:
        # Try to get RAPTOR summaries collection
        collection = chroma_client.get_collection("raptor")
        
        results = collection.query(
            query_texts=[query],
            n_results=k
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        # Convert to Documents
        docs = []
        for i, doc_text in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
            metadata['source'] = 'raptor_summary'
            metadata['level'] = metadata.get('level', 1)
            
            docs.append(Document(
                page_content=doc_text,
                metadata=metadata
            ))
        
        return docs
    
    except Exception as e:
        print(f"⚠️ RAPTOR collection not found or error: {e}")
        return []  # Return empty if RAPTOR not built yet
