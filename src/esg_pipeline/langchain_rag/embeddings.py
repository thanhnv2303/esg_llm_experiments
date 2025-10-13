from __future__ import annotations

import hashlib
from typing import Iterable, List

import numpy as np
from langchain_core.embeddings import Embeddings


class HashingEmbeddings(Embeddings):
    """Deterministic hashing-based embeddings for lightweight experiments."""

    def __init__(self, dimension: int = 768) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dimension
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in _tokenise(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimension
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0.0:
            vector /= norm
        return vector.astype(np.float32).tolist()


def _tokenise(text: str) -> Iterable[str]:
    current: List[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            yield "".join(current)
            current.clear()
    if current:
        yield "".join(current)


__all__ = ["HashingEmbeddings"]
