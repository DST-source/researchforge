"""
Hybrid Retriever: Combines vector search + keyword search (BM25)
"""
from typing import List, Dict
import chromadb
from langchain.schema import Document
from rank_bm25 import BM25Okapi


class HybridRetriever:
    """Hybrid retrieval combining semantic + keyword search"""
    
    def __init__(self, chroma_client: chromadb.HttpClient):
        self.chroma = chroma_client
        try:
            self.collection = chroma_client.get_collection("chunks")
        except:
            print("⚠️ 'chunks' collection not found, using 'documents'")
            self.collection = chroma_client.get_collection("documents")
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Hybrid retrieval: vector search + BM25
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        try:
            # Vector search
            results = self.collection.query(
                query_texts=[query],
                n_results=k * 2  # Get more for re-ranking
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            # Convert to Document objects
            docs = []
            for i, doc_text in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                docs.append(Document(
                    page_content=doc_text,
                    metadata=metadata
                ))
            
            # BM25 keyword search (simple implementation)
            query_terms = query.lower().split()
            corpus = [doc.page_content.lower().split() for doc in docs]
            
            if corpus:
                bm25 = BM25Okapi(corpus)
                bm25_scores = bm25.get_scores(query_terms)
                
                # Combine scores (vector already sorted, add BM25 boost)
                for i, doc in enumerate(docs):
                    doc.metadata['bm25_score'] = bm25_scores[i]
                    doc.metadata['hybrid_score'] = (k - i) + bm25_scores[i]
                
                # Re-sort by hybrid score
                docs.sort(key=lambda x: x.metadata.get('hybrid_score', 0), reverse=True)
            
            return docs[:k]
        
        except Exception as e:
            print(f"⚠️ Hybrid retrieval failed: {e}")
            return []
