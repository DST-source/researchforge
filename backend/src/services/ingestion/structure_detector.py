"""
Structure Detector: Extract PDF structure (TOC, sections)
Optional for Phase 0
"""
import fitz
from typing import List, Dict


def detect_structure(pdf_path: str) -> Dict:
    """
    Detect PDF structure (table of contents, sections)
    
    Returns:
        {"sections": List[Dict], "toc": List[str]}
    """
    doc = fitz.open(pdf_path)
    
    # Try to extract TOC
    toc = doc.get_toc()
    
    sections = []
    if toc:
        for level, title, page_num in toc:
            sections.append({
                "title": title,
                "level": level,
                "page": page_num
            })
    
    doc.close()
    
    return {
        "sections": sections,
        "toc": [s["title"] for s in sections]
    }


if __name__ == "__main__":
    print("✅ Structure detector loaded (optional)")
