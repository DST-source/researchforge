"""
Semantic Chunking (Lance Martin Notebooks 1-4)
Split text into meaningful chunks respecting sentence boundaries
"""
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class SemanticChunker:
    """
    Semantic text chunking with sentence awareness
    Based on Lance Martin's RAG-from-scratch Parts 1-4
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # RecursiveCharacterTextSplitter respects sentence boundaries
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Split text into semantic chunks
        
        Args:
            text: Full paper text
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Document objects
        """
        if not text or len(text.strip()) < 50:
            return []
        
        # Split text
        chunks = self.splitter.split_text(text)
        
        # Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata['chunk_index'] = i
            doc_metadata['char_count'] = len(chunk)
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        return documents


# Quick test
if __name__ == "__main__":
    chunker = SemanticChunker(chunk_size=512, overlap=50)
    test_text = "This is a test sentence. " * 100
    chunks = chunker.chunk(test_text)
    print(f"✅ Created {len(chunks)} chunks from {len(test_text)} chars")
    print(f"   First chunk: {chunks[0].page_content[:100]}...")
