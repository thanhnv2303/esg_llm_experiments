from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List

import numpy as np
from langchain_core.embeddings import Embeddings

from .config import LangchainRAGConfig

try:  # pragma: no cover - optional dependency
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    try:
        from langchain_community.embeddings import (
            HuggingFaceEmbeddings as _DeprecatedHuggingFaceEmbeddings,
        )  # type: ignore[assignment]
    except ImportError:  # pragma: no cover - optional dependency
        HuggingFaceEmbeddings = None  # type: ignore[assignment]
    else:  # pragma: no cover - optional dependency
        HuggingFaceEmbeddings = _DeprecatedHuggingFaceEmbeddings  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # pragma: no cover - optional dependency
    OpenAIEmbeddings = None  # type: ignore[assignment]


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


def create_embeddings(config: LangchainRAGConfig) -> Embeddings:
    """Factory that builds an embedding backend based on configuration."""

    backend = (config.embedding_backend or "hash").lower().strip()
    if backend in {"hash", "hashing"}:
        return HashingEmbeddings(dimension=config.embedding_dimension)

    kwargs: Dict[str, object] = dict(config.embedding_kwargs)

    if backend in {"huggingface", "hf", "sentence-transformers"}:
        if HuggingFaceEmbeddings is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "HuggingFace embeddings requested but 'langchain-community' did not "
                "have the sentence-transformers extra installed."
            )
        model_name = config.embedding_model or "sentence-transformers/all-mpnet-base-v2"
        return HuggingFaceEmbeddings(model_name=model_name, **kwargs)

    if backend in {"openai", "openai-embeddings", "azure-openai"}:
        if OpenAIEmbeddings is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "OpenAI embeddings requested but 'langchain-openai' is not installed."
            )
        model_name = config.embedding_model or "text-embedding-3-small"
        kwargs.setdefault("model", model_name)
        return OpenAIEmbeddings(**kwargs)

    raise ValueError(f"Unsupported embedding backend '{config.embedding_backend}'.")


__all__ = ["HashingEmbeddings", "create_embeddings"]
