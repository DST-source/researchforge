"""
Multimodal Image Extraction (from your earlier code)
Uses PyMuPDF to extract images with metadata
"""
import fitz
from typing import List, Dict


def extract_images(pdf_path: str, max_images: int = 5) -> List[Dict]:
    """
    Extract images from PDF using PyMuPDF
    
    Args:
        pdf_path: Path to PDF file
        max_images: Max images to extract per paper
        
    Returns:
        List of dicts with {bytes, page, source}
    """
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        if len(images) >= max_images:
            break
            
        page = doc[page_num]
        
        for img_idx, img in enumerate(page.get_images()):
            if len(images) >= max_images:
                break
                
            try:
                xref = img[0]
                img_data = doc.extract_image(xref)
                
                images.append({
                    'bytes': img_data["image"],
                    'page': page_num + 1,
                    'source': pdf_path.split('/')[-1]
                })
            except Exception as e:
                print(f"⚠️  Image extraction failed: {e}")
                continue
    
    doc.close()
    return images
