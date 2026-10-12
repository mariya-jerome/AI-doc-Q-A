import faiss
import numpy as np
from typing import List, Dict

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        
    def build_index(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Builds a FAISS index from the provided embeddings and stores the chunks.
        Using IndexFlatIP for Inner Product since embeddings are normalized, 
        which is equivalent to cosine similarity.
        """
        if len(embeddings) == 0:
            raise ValueError("No embeddings to index.")
            
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.chunks = chunks

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Dict]:
        """
        Searches the index for the most similar chunks to the query_embedding.
        """
        if self.index is None:
            raise RuntimeError("Index has not been built yet.")
            
        # FAISS expects a 2D array
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.chunks):
                results.append(self.chunks[idx])
                
        return results
