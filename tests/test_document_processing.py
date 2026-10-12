import pytest
from unittest.mock import patch, MagicMock
from document_processor import extract_text_from_pdf
import numpy as np

# 1. A PDF with extractable text.
def test_extract_text_with_text():
    with patch("document_processor.PdfReader") as MockReader:
        mock_instance = MockReader.return_value
        mock_instance.is_encrypted = False
        
        page1 = MagicMock()
        page1.extract_text.return_value = "Hello World"
        
        mock_instance.pages = [page1]
        
        result = extract_text_from_pdf("dummy.pdf")
        assert len(result) == 1
        assert result[0]["text"] == "Hello World"

# 2. A PDF with no extractable text.
def test_extract_text_no_text():
    with patch("document_processor.PdfReader") as MockReader:
        mock_instance = MockReader.return_value
        mock_instance.is_encrypted = False
        
        page1 = MagicMock()
        page1.extract_text.return_value = "" # No text
        mock_instance.pages = [page1]
        
        with pytest.raises(ValueError, match="no extractable text"):
            extract_text_from_pdf("dummy.pdf")

# 6. Creation of embeddings with the expected dimensions
@patch("embeddings.SentenceTransformer")
def test_embeddings_dimensions(mock_st):
    from embeddings import generate_embeddings
    import embeddings
    embeddings._embedding_model = None
    
    mock_model = MagicMock()
    # Return fake 384-dimensional embeddings for 2 texts
    mock_model.encode.return_value = np.zeros((2, 384))
    mock_st.return_value = mock_model
    
    res = generate_embeddings(["Text 1", "Text 2"])
    assert res.shape == (2, 384)

# 7. FAISS retrieval of relevant chunks
def test_faiss_retrieval():
    from vector_store import VectorStore
    vs = VectorStore()
    
    # 2 embeddings of dimension 2 for simplicity
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype='float32')
    chunks = [
        {"text": "Horizontal", "page_number": 1},
        {"text": "Vertical", "page_number": 2}
    ]
    
    vs.build_index(embeddings, chunks)
    
    # search for [1.0, 0.0]
    query_emb = np.array([[0.9, 0.1]], dtype='float32')
    results = vs.search(query_emb, top_k=1)
    
    assert len(results) == 1
    assert results[0]["text"] == "Horizontal"
