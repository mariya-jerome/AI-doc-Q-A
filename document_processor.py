from pypdf import PdfReader
from typing import List, Dict, Optional

def extract_text_from_pdf(file) -> List[Dict]:
    """
    Extracts text from a PDF file page by page.
    Returns a list of dictionaries containing page number and text.
    """
    pages_data = []
    try:
        reader = PdfReader(file)
        if reader.is_encrypted:
            raise ValueError("PDF is password-protected and cannot be read.")
            
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # Remove unnecessary whitespace
                text = " ".join(text.split())
                if text.strip():
                    pages_data.append({
                        "page_number": i + 1,
                        "text": text
                    })
        
        if not pages_data:
            raise ValueError("The PDF contains no extractable text.")
            
        return pages_data
    except Exception as e:
        raise ValueError(f"Failed to extract pdf: {e}")

def create_chunks(pages_data: List[Dict], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Dict]:
    """
    Splits text into overlapping chunks while preserving page numbers.
    """
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap.")
        
    chunks = []
    
    for page in pages_data:
        text = page["text"]
        page_number = page["page_number"]
        
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page_number": page_number
                })
            
            start += (chunk_size - chunk_overlap)
            
    return chunks
