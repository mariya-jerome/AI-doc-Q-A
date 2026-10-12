import pytest
from document_processor import create_chunks

def test_create_chunks_splitting():
    pages_data = [
        {"page_number": 1, "text": "This is a test document page that will be split into multiple smaller chunks for embedding generation."}
    ]
    # Small chunk size to force splitting
    chunks = create_chunks(pages_data, chunk_size=30, chunk_overlap=10)
    assert len(chunks) > 1
    assert chunks[0]["page_number"] == 1

def test_create_chunks_overlap():
    pages_data = [
        {"page_number": 1, "text": "abcdefghijklmnopqrstuvwxyz"}
    ]
    chunks = create_chunks(pages_data, chunk_size=10, chunk_overlap=5)
    # Chunk 0: [0:10] -> "abcdefghij"
    # Step = 5
    # Chunk 1: [5:15] -> "fghijklmnopqrst"
    assert chunks[0]["text"] == "abcdefghij"
    assert chunks[1]["text"] == "fghijklmno"

def test_create_chunks_preserve_page_number():
    pages_data = [
        {"page_number": 1, "text": "Page one text content here."},
        {"page_number": 2, "text": "Page two text content here."}
    ]
    chunks = create_chunks(pages_data, chunk_size=50, chunk_overlap=10)
    assert chunks[0]["page_number"] == 1
    assert chunks[1]["page_number"] == 2
