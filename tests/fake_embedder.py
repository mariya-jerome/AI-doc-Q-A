"""Deterministic hashing embedder so retrieval tests do not download a model."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray


class HashingEmbedder:
    """Map tokens into a fixed-size bag-of-words vector, then L2-normalize."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> NDArray[np.float32]:
        del convert_to_numpy, show_progress_bar
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.md5(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                matrix[row, index] += 1.0
            if not text.lower().split():
                matrix[row, 0] = 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        return matrix
