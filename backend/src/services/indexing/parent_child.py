"""
Multi-Representation Indexing (Lance Martin Notebook 12)
Create parent-child chunk relationships
"""
from typing import List
from langchain.schema import Document


def create_parent_child_chunks(
    chunks: List[Document],
    parent_size: int = 1500,
    chunks_per_parent: int = 3
) -> List[Document]:
    """
    Create parent chunks from child chunks
    
    Strategy:
    - Small chunks (512 chars) for precise retrieval
    - Large parents (1500 chars) for rich context
    - Link via parent_id in metadata
    
    Args:
        chunks: List of child chunks (512 chars)
        parent_size: Target parent size
        chunks_per_parent: How many children per parent
        
    Returns:
        List of parent Document objects
    """
    parents = []
    
    # Group chunks into parents
    for i in range(0, len(chunks), chunks_per_parent):
        group = chunks[i:i + chunks_per_parent]
        
        # Combine chunk text into parent
        parent_text = "\n\n".join([c.page_content for c in group])
        
        # Create parent ID
        parent_id = f"parent_{i // chunks_per_parent}"
        
        # Update child metadata with parent_id
        for child in group:
            child.metadata['parent_id'] = parent_id
        
        # Create parent document
        parent_meta = group[0].metadata.copy()
        parent_meta['type'] = 'parent'
        parent_meta['child_count'] = len(group)
        parent_meta['char_count'] = len(parent_text)
        
        parents.append(Document(
            page_content=parent_text[:parent_size],  # Truncate to target
            metadata=parent_meta
        ))
    
    return parents


# Test
if __name__ == "__main__":
    from semantic_chunker import SemanticChunker
    
    chunker = SemanticChunker()
    test_text = "Sentence " * 500
    chunks = chunker.chunk(test_text)
    
    parents = create_parent_child_chunks(chunks)
    print(f"✅ {len(chunks)} chunks → {len(parents)} parents")
    print(f"   Parent 0 has child chunks: {[c.metadata.get('parent_id') for c in chunks[:3]]}")
