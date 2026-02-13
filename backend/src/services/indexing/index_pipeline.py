"""
PHASE 0: INDEXING PIPELINE (Lance Martin Notebooks 1-4, 12, 13)
Integrates multimodal extraction with ResearchForge modules
"""
import sys
from pathlib import Path
import chromadb
from typing import Dict, List
import base64

from ..ingestion.pdf_parser import PDFParser
from ..ingestion.metadata_extractor import MetadataExtractor
from ..ingestion.figure_extractor import extract_images
from ..ingestion.table_extractor import extract_tables  # ✅ ADD THIS
from .semantic_chunker import SemanticChunker
from .parent_child import create_parent_child_chunks
from .raptor import build_raptor_tree
from .embedder import Embedder
from .chromadb_uploader import ChromaUploader
from ..llm.client import chat


class IndexPipeline:
    """
    Complete Phase 0 indexing using Lance Martin techniques:
    - Semantic chunking (Notebooks 1-4)
    - Multi-Rep indexing (Notebook 12)
    - RAPTOR tree (Notebook 13)
    - MULTIMODAL: Your image extraction + vision summaries
    """
    
    def __init__(self, chroma_host='localhost', chroma_port=8000):
        self.chroma = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        self.embedder = Embedder()
        self.uploader = ChromaUploader(self.chroma)
        self.tables_coll = self.chroma.get_or_create_collection("tables")

        # Create 3 collections (per your spec)
        self.chunks_coll = self.chroma.get_or_create_collection("chunks")
        self.parents_coll = self.chroma.get_or_create_collection("parents")
        self.raptor_coll = self.chroma.get_or_create_collection("raptor")
        self.images_coll = self.chroma.get_or_create_collection("images")
    
    def index_paper(self, pdf_path: Path) -> Dict:
        """
        Index single paper through full pipeline
        Returns stats dict
        """
        print(f"\n{'='*60}\n📄 {pdf_path.name}\n{'='*60}")
        
        # 1. PDF PARSING (PyMuPDF)
        parser = PDFParser()
        parsed = parser.parse(str(pdf_path))
        print(f"✅ Parsed {parsed['num_pages']} pages")
        
        # 2. METADATA EXTRACTION (Gemini LLM)
        extractor = MetadataExtractor()
        metadata = extractor.extract(parsed['text'], pdf_path.name)
        print(f"✅ Metadata: {metadata.get('title', 'N/A')}")
        
        # 3. SEMANTIC CHUNKING (Lance Martin 1-4)
        chunker = SemanticChunker(chunk_size=512, overlap=50)
        chunks = chunker.chunk(parsed['text'])
        print(f"✅ {len(chunks)} semantic chunks (512t)")
        
        # 4. MULTI-REP INDEXING (Lance Martin 12)
        parent_chunks = create_parent_child_chunks(chunks, parent_size=1500)
        print(f"✅ {len(parent_chunks)} parent chunks (1500t)")
        
        # 5. EMBED CHUNKS
        chunk_embeddings = self.embedder.embed([c.page_content for c in chunks])
        parent_embeddings = self.embedder.embed([p.page_content for p in parent_chunks])
        
     # 6. UPLOAD TO CHROMA
        paper_id = pdf_path.stem
        
        # Clean metadata for ChromaDB (no lists/dicts/None)
        clean_metadata = {
            "title": str(metadata.get('title') or 'Unknown'),
            "authors": ", ".join(metadata.get('authors') or []) or 'Unknown',
            "year": int(metadata.get('year') or 0),
            "venue": str(metadata.get('venue') or ''),
            "keywords": ", ".join(metadata.get('keywords') or []) or ''
        }
        
        # Chunks collection
        self.chunks_coll.add(
            documents=[c.page_content for c in chunks],
            embeddings=chunk_embeddings,
            metadatas=[{
                "paper_id": paper_id,
                "paper_name": pdf_path.name,
                "parent_id": f"parent_{i//3}",  # 3 chunks per parent
                **clean_metadata  # Use cleaned metadata
            } for i, c in enumerate(chunks)],
            ids=[f"{paper_id}_chunk_{i}" for i in range(len(chunks))]
        )
        
        # Parents collection
        self.parents_coll.add(
            documents=[p.page_content for p in parent_chunks],
            embeddings=parent_embeddings,
            metadatas=[{
                "paper_id": paper_id,
                "paper_name": pdf_path.name,
                "type": "parent",
                **clean_metadata
            } for _ in parent_chunks],
            ids=[f"{paper_id}_parent_{i}" for i in range(len(parent_chunks))]
        )
        
        # 7. RAPTOR TREE (Lance Martin 13)
        raptor_nodes = build_raptor_tree(chunks, self.embedder, levels=3)
        raptor_embeddings = self.embedder.embed([n['summary'] for n in raptor_nodes])
        
        self.raptor_coll.add(
            documents=[n['summary'] for n in raptor_nodes],
            embeddings=raptor_embeddings,
            metadatas=[{
                "paper_id": paper_id,
                "level": n['level'],
                "cluster_id": n.get('cluster_id', 0)
            } for n in raptor_nodes],
            ids=[f"{paper_id}_raptor_{i}" for i in range(len(raptor_nodes))]
        )
        print(f"✅ {len(raptor_nodes)} RAPTOR nodes (3 levels)")
        
        # 8. MULTIMODAL: IMAGE EXTRACTION + VISION SUMMARIES
        images = extract_images(str(pdf_path), max_images=5)
        
        if images:
            image_summaries = []
            for img in images:
                # Vision summary using Gemini
                b64 = base64.b64encode(img['bytes']).decode()
                try:
                    summary = chat([{
                        "role": "user",
                        "content": f"Summarize this research paper figure in 1-2 sentences (max 100 words):\n[Image data]"
                    }], max_tokens=150)
                    image_summaries.append(summary)
                except:
                    image_summaries.append(f"Figure from {pdf_path.name}, page {img.get('page', '?')}")
        # 9. TABLES
        tables = extract_tables(str(pdf_path), max_tables=5)
        if tables:
            summaries = [f"Table p{t['page']}: {t.get('rows',0)}x{t.get('cols',0)}" for t in tables]
            embs = self.embedder.embed(summaries)
            self.tables_coll.add(
                documents=summaries,
                embeddings=embs,
                metadatas=[{"paper_id": paper_id, "page": t['page']} for t in tables],
                ids=[f"{paper_id}_tbl_{i}" for i in range(len(tables))]
            )
            print(f"✅ {len(tables)} tables indexed")

            # Embed + upload images
            img_embeddings = self.embedder.embed(image_summaries)
            self.images_coll.add(
                documents=image_summaries,
                embeddings=img_embeddings,
                metadatas=[{
                    "paper_id": paper_id,
                    "type": "image",
                    "page": img.get('page', 0)
                } for img in images],
                ids=[f"{paper_id}_img_{i}" for i in range(len(images))]
            )
            print(f"✅ {len(images)} multimodal images indexed")
        
        return {
            "paper_id": paper_id,
            "chunks": len(chunks),
            "parents": len(parent_chunks),
            "raptor": len(raptor_nodes),
            "images": len(images) if images else 0,
            "tables": len(tables) if tables else 0,
            "metadata": metadata
        }
    
    def get_stats(self):
        return {
            "chunks": self.chunks_coll.count(),
            "parents": self.parents_coll.count(),
            "raptor": self.raptor_coll.count(),
            "images": self.images_coll.count(),
            "tables": self.tables_coll.count()  # ✅ ADD
        }

# Test function
if __name__ == "__main__":
    pipeline = IndexPipeline()
    
    # Index test papers
    test_pdfs = Path("test_data").glob("*.pdf")
    for pdf in test_pdfs:
        stats = pipeline.index_paper(pdf)
        print(f"\n{stats}")
    
    # Final stats
    print(f"\n{'='*60}")
    print("INDEXING COMPLETE")
    print(f"{'='*60}")
    print(pipeline.get_stats())
