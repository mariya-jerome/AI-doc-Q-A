import pytest
from unittest.mock import patch, MagicMock
from rag_pipeline import construct_prompt, generate_answer

# 8. Construction of a grounded RAG prompt.
def test_construct_prompt():
    query = "What is the capital?"
    context_chunks = [
        {"text": "Paris is the capital of France.", "page_number": 1}
    ]
    
    system_prompt, source_pages = construct_prompt(query, context_chunks)
    assert "Paris is the capital of France." in system_prompt
    assert source_pages == [1]
    assert "not found in the uploaded document" in system_prompt

# 9. Handling of an empty question.
def test_empty_question():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        generate_answer("", MagicMock())

# 10. Handling of a missing API key.
def test_missing_api_key():
    with patch("config.GROQ_API_KEY", ""):
        with pytest.raises(ValueError, match="GROQ_API_KEY is missing"):
            generate_answer("Valid query", MagicMock(), api_key="")

# 11. Handling of an API failure / decommissioned model / invalid key.
def test_api_failure():
    with patch("rag_pipeline.generate_query_embedding") as mock_emb:
        mock_emb.return_value = MagicMock()
        mock_vs = MagicMock()
        mock_vs.search.return_value = [{"text": "Context", "page_number": 1}]
        
        with patch("rag_pipeline.Groq") as mock_groq:
            mock_client = mock_groq.return_value
            mock_client.chat.completions.create.side_effect = Exception("400 model_decommissioned")
            
            with pytest.raises(ValueError, match="decommissioned by Groq"):
                generate_answer("query", mock_vs, api_key="dummy_key")

# 12. Handling of a question whose answer is not present in the retrieved context.
def test_answer_not_in_context():
    with patch("rag_pipeline.generate_query_embedding") as mock_emb:
        mock_emb.return_value = MagicMock()
        mock_vs = MagicMock()
        mock_vs.search.return_value = [{"text": "Apples are red.", "page_number": 1}]
        
        with patch("rag_pipeline.Groq") as mock_groq:
            mock_client = mock_groq.return_value
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Information not found in the uploaded document."
            mock_client.chat.completions.create.return_value = mock_response
            
            answer, sources = generate_answer("What color is the sky?", mock_vs, api_key="dummy_key")
            assert "not found" in answer.lower()
