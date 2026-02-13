"""
PDF Parser using PyMuPDF
FIXED: Returns actual text, not "--- Page N ---"
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict


class PDFParser:
    """Parse PDF to extract text"""
    
    def parse(self, pdf_path: str) -> Dict:
        """Extract text from PDF"""
        doc = fitz.open(pdf_path)
        
        # Extract ALL text from ALL pages
        full_text = ""
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            full_text += page_text + "\n"
        
        return {
            "text": full_text,
            "num_pages": len(doc),
            "metadata": {"filename": Path(pdf_path).name}
        }


# Test
if __name__ == "__main__":
    parser = PDFParser()
    result = parser.parse("test_data/attention.pdf")
    print(f"✅ Extracted {len(result['text'])} chars from {result['num_pages']} pages")
    print(f"First 200 chars: {result['text'][:200]}")
