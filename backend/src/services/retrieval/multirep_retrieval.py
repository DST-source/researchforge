"""
Multi-Rep Retrieval: Expand child chunks to parent chunks (Lance Martin 12)
"""
from typing import List
import chromadb
from langchain.schema import Document


class MultiRepRetriever:
    """Expand retrieved chunks to their parent contexts"""
    
    def __init__(self, chroma_client: chromadb.HttpClient):
        self.chroma = chroma_client
        self.chunks_coll = chroma_client.get_collection("chunks")
        self.parents_coll = chroma_client.get_collection("parents")
    
    def expand_to_parents(self, child_docs: List[Document]) -> List[Document]:
        """Expand child chunks to parent chunks"""
        parent_ids = set()
        parent_docs = []
        
        # Extract parent IDs from children
        for doc in child_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in parent_ids:
                parent_ids.add(parent_id)
        
        # Fetch parents from ChromaDB
        if parent_ids:
            results = self.parents_coll.get(ids=list(parent_ids))
            
            for i, doc_text in enumerate(results["documents"]):
                parent_docs.append(Document(
                    page_content=doc_text,
                    metadata=results["metadatas"][i]
                ))
        
        return parent_docs
