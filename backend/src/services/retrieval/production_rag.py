"""
ProductionRAG: Complete 4-Phase Pipeline
Orchestrates all Lance Martin techniques
"""
import chromadb
from typing import Dict, List
from langchain.schema import Document
from langsmith import traceable

# Import tracing utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.tracing import trace_phase, trace_retrieval, trace_llm, trace_tool, trace_component

# Phase 1
from .multi_query import generate_multi_queries
from .hyde import generate_hyde_document

# Phase 2
from .hybrid_retriever import HybridRetriever
from .multirep_retrieval import MultiRepRetriever
from .raptor_traverser import query_raptor_tree
from .crag import CRAG

# Phase 3
from .rag_fusion import reciprocal_rank_fusion
from .reranker import rerank_documents
from .contect_compressor import compress_context

# Phase 4
from .self_rag import SelfRAG
from ..llm.answer_generator import AnswerGenerator
"""
ProductionRAG: Complete 4-Phase Pipeline
Orchestrates all Lance Martin techniques
"""
import chromadb
from typing import Dict, List
from langchain.schema import Document
from langsmith import traceable

# Import tracing utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.tracing import trace_phase, trace_retrieval, trace_llm, trace_tool, trace_component

# Phase 1
from .multi_query import generate_multi_queries
from .hyde import generate_hyde_document

# Phase 2
from .hybrid_retriever import HybridRetriever
from .multirep_retrieval import MultiRepRetriever
from .raptor_traverser import query_raptor_tree
from .crag import CRAG

# Phase 3
from .rag_fusion import reciprocal_rank_fusion
from .reranker import rerank_documents
from .contect_compressor import compress_context

# Phase 4
from .self_rag import SelfRAG
from ..llm.answer_generator import AnswerGenerator


class ProductionRAG:
    """Complete RAG pipeline with LangSmith tracing"""
    
    def __init__(self, chroma_host='localhost', chroma_port=8000):
        self.chroma = chromadb.HttpClient(chroma_host, chroma_port)
        
        # Initialize components
        self.hybrid_retriever = HybridRetriever(self.chroma)
        self.multirep = MultiRepRetriever(self.chroma)
        self.crag = CRAG()
        self.self_rag = SelfRAG()
        self.generator = AnswerGenerator()
    
    @traceable(
        name="ProductionRAG Pipeline",
        run_type="chain",
        tags=["rag", "production", "full-pipeline"],
        metadata={"version": "1.0", "model": "gemini-2.5-flash"}
    )
    def answer_question(self, question: str, top_k: int = 5) -> Dict:
        """Complete RAG pipeline: question → answer with citations"""
        
        print(f"\n{'='*60}")
        print(f"❓ QUESTION: {question}")
        print(f"{'='*60}")
        
        # PHASE 1: Query Construction
        docs_phase1 = self._phase1_query_construction(question)
        
        # PHASE 2: Retrieval
        docs_phase2 = self._phase2_retrieval(question, docs_phase1)
        
        # PHASE 3: Post-Retrieval
        docs_phase3 = self._phase3_post_retrieval(question, docs_phase2)
        
        # PHASE 4: Generation
        result = self._phase4_generation(question, docs_phase3, top_k)
        
        print(f"\n{'='*60}")
        print(f"✅ ANSWER: {result['answer'][:200]}...")
        print(f"📚 CITATIONS: {result['citations']}")
        print(f"{'='*60}\n")
        
        return result
    
    @trace_phase("Query Construction", 1)
    def _phase1_query_construction(self, question: str) -> List[List[Document]]:
        """Phase 1: Multi-query + HyDE"""
        print("\n📋 PHASE 1: Query Construction...")
        
        # Multi-query
        multi_queries = self._multi_query(question)
        
        # HyDE
        hyde_doc = self._hyde(question)
        
        all_queries = multi_queries + [hyde_doc]
        print(f"  ✓ Generated {len(all_queries)} query variants")
        
        # Retrieve for each query
        all_docs = []
        for query in all_queries:
            docs = self._basic_retrieve(query, k=5)
            all_docs.append(docs)
        
        return all_docs
    
    @trace_retrieval("Multi-Query Generation")
    def _multi_query(self, question: str) -> List[str]:
        """Generate multiple query variants"""
        return generate_multi_queries(question, n=3)
    
    @trace_llm("HyDE Generation")
    def _hyde(self, question: str) -> str:
        """Generate hypothetical document"""
        return generate_hyde_document(question)
    
    @trace_retrieval("Basic Vector Search")
    def _basic_retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Basic retrieval from chunks collection"""
        return self.hybrid_retriever.retrieve(query, k=k)
    
    @trace_phase("Retrieval", 2)
    def _phase2_retrieval(self, question: str, docs_lists: List[List[Document]]) -> List[Document]:
        """Phase 2: Hybrid + Multi-Rep + RAPTOR"""
        print("\n🔍 PHASE 2: Hybrid Retrieval...")
        
        # Flatten all docs
        child_docs = [doc for doc_list in docs_lists for doc in doc_list]
        
        # Multi-Rep expansion
        parent_docs = self._multirep_expand(child_docs[:10])
        print(f"  ✓ Retrieved {len(child_docs)} chunks → {len(parent_docs)} parents")
        
        # RAPTOR summaries
        raptor_docs = self._raptor_retrieve(question)
        
        # Combine
        all_retrieved = child_docs + parent_docs + raptor_docs
        
        # CRAG check
        if not self.crag.check_relevance(all_retrieved, question):
            web_docs = self._crag_fallback(question)
            all_retrieved.extend(web_docs)
        
        return all_retrieved
    
    @trace_retrieval("Multi-Rep Expansion")
    def _multirep_expand(self, child_docs: List[Document]) -> List[Document]:
        """Expand children to parents"""
        return self.multirep.expand_to_parents(child_docs)
    
    @trace_retrieval("RAPTOR Tree Query")
    def _raptor_retrieve(self, question: str) -> List[Document]:
        """Query RAPTOR tree"""
        return query_raptor_tree(self.chroma, question, k=3)
    
    @trace_tool("CRAG Web Fallback")
    def _crag_fallback(self, question: str) -> List[Document]:
        """CRAG web fallback"""
        return self.crag.fallback_retrieve(question)
    
    @trace_phase("Post-Retrieval", 3)
    def _phase3_post_retrieval(self, question: str, docs: List[Document]) -> List[Document]:
        """Phase 3: RAG Fusion + Reranking + Compression"""
        print("\n⚙️ PHASE 3: Fusion + Reranking...")
        
        # RAG Fusion (RRF)
        fused_docs = self._rag_fusion(docs)
        
        # Rerank
        reranked_docs = self._rerank(question, fused_docs[:20])
        print(f"  ✓ Reranked to top {len(reranked_docs)} docs")
        
        # Compress
        final_docs = self._compress(reranked_docs)
        print(f"  ✓ Compressed to {len(final_docs)} final docs")
        
        return final_docs
    
    @trace_tool("RAG Fusion (RRF)")
    def _rag_fusion(self, docs: List[Document]) -> List[Document]:
        """Apply RRF fusion"""
        # Split into 3 lists for RRF
        n = len(docs) // 3
        return reciprocal_rank_fusion([docs[:n], docs[n:2*n], docs[2*n:]])
    
    @trace_component("Reranking", "reranker")
    def _rerank(self, question: str, docs: List[Document]) -> List[Document]:
        """Rerank documents"""
        return rerank_documents(question, docs)
    
    @trace_tool("Context Compression")
    def _compress(self, docs: List[Document]) -> List[Document]:
        """Compress context"""
        return compress_context(docs, max_tokens=2000)
    
    @trace_phase("Generation", 4)
    def _phase4_generation(self, question: str, docs: List[Document], top_k: int) -> Dict:
        """Phase 4: Self-RAG + Answer Generation"""
        print("\n✍️ PHASE 4: Answer Generation...")
        
        # Self-RAG grading
        graded_docs = self._self_rag_grade(question, docs)
        
        # Generate answer
        result = self._generate_answer(question, graded_docs[:top_k])
        
        return result
    
    @trace_component("Self-RAG Grading", "llm")
    def _self_rag_grade(self, question: str, docs: List[Document]) -> List[Document]:
        """Grade document relevance"""
        return self.self_rag.grade_documents(docs, question)
    
    @trace_llm("Answer Generation")
    def _generate_answer(self, question: str, docs: List[Document]) -> Dict:
        """Generate final answer with citations"""
        return self.generator.generate(question, docs)


# Test
if __name__ == "__main__":
    rag = ProductionRAG()
    result = rag.answer_question("What is the Transformer architecture?")
    print(f"\n✅ Full Answer:\n{result['answer']}")
