"""
Self-RAG: Reflection on retrieved context (Lance Martin Notebook 15)
"""
from typing import List
from langchain.schema import Document
from ..llm.client import chat


class SelfRAG:
    """Self-RAG: Grade relevance + reflect on answer quality"""
    
    def grade_documents(self, docs: List[Document], question: str) -> List[Document]:
        """Grade each document for relevance"""
        relevant_docs = []
        
        for doc in docs:
            prompt = f"""Grade the relevance of this document to the question.

Question: {question}

Document: {doc.page_content[:500]}

Is this document relevant? Answer ONLY 'yes' or 'no'."""
            
            try:
                grade = chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10
                ).strip().lower()
                
                if "yes" in grade:
                    relevant_docs.append(doc)
            except:
                # If grading fails, keep the document
                relevant_docs.append(doc)
        
        return relevant_docs if relevant_docs else docs[:5]  # Fallback
