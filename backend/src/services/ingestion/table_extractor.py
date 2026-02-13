"""
Table Extraction from PDFs
Uses pdfplumber for structure detection
"""
import pdfplumber
from typing import List, Dict


def extract_tables(pdf_path: str, max_tables: int = 5) -> List[Dict]:
    """
    Extract tables from PDF
    
    Args:
        pdf_path: Path to PDF
        max_tables: Max tables to extract
        
    Returns:
        List of dicts: {"page": int, "data": List[List], "caption": str}
    """
    tables = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                if len(tables) >= max_tables:
                    break
                
                # Extract tables from page
                page_tables = page.extract_tables()
                
                for table in page_tables:
                    if len(tables) >= max_tables:
                        break
                    
                    if table and len(table) > 1:  # Has rows
                        tables.append({
                            "page": page_num + 1,
                            "data": table,
                            "caption": f"Table {len(tables) + 1}",
                            "rows": len(table),
                            "cols": len(table[0]) if table else 0
                        })
    except Exception as e:
        print(f"⚠️ Table extraction failed: {e}")
    
    return tables


# Test
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        tables = extract_tables(sys.argv[1])
        print(f"✅ Extracted {len(tables)} tables")
        for t in tables:
            print(f"   Page {t['page']}: {t['rows']}x{t['cols']}")
