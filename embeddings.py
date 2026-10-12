from sentence_transformers import SentenceTransformer
import config
from typing import List
import numpy as np

# Global variable to reuse the loaded model
_embedding_model = None

def get_model():
    """
    Loads the embedding model once and reuses it.
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
    return _embedding_model

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generates normalized embeddings for a list of texts.
    Returns a numpy array of embeddings.
    """
    model = get_model()
    try:
        # Generate embeddings
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings: {e}")

def generate_query_embedding(query: str) -> np.ndarray:
    """
    Generates a normalized embedding for a single query.
    Return a numpy array.
    """
    model = get_model()
    try:
        embedding = model.encode([query], normalize_embeddings=True)
        return embedding
    except Exception as e:
        raise RuntimeError(f"Failed to generate query embedding: {e}")
