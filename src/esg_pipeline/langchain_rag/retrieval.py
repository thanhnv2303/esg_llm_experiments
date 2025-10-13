from __future__ import annotations

import textwrap
from typing import Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from .config import LangchainRAGConfig
from .documents import page_one_based
from .embeddings import HashingEmbeddings
from .types import RetrievedChunk


def build_vector_store(documents: Sequence[Document], config: LangchainRAGConfig) -> FAISS:
    embeddings = HashingEmbeddings(dimension=config.embedding_dimension)
    return FAISS.from_documents(documents, embeddings)


def retrieve_chunks(vector_store: FAISS, query: str, config: LangchainRAGConfig) -> List[RetrievedChunk]:
    if not query.strip():
        return []

    if config.use_mmr:
        base_docs = vector_store.max_marginal_relevance_search(
            query,
            k=config.top_k,
            fetch_k=max(config.mmr_fetch_k, config.top_k),
            lambda_mult=config.mmr_lambda,
        )
        doc_score_pairs: Iterable[Tuple[Document, Optional[float]]] = (
            (doc, None) for doc in base_docs
        )
    else:
        base_docs = vector_store.similarity_search_with_score(query, k=config.top_k)
        doc_score_pairs = base_docs

    return [chunk_from_document(doc, score) for doc, score in doc_score_pairs]


def chunk_from_document(doc: Document, score: Optional[float]) -> RetrievedChunk:
    metadata = dict(doc.metadata)
    page = page_one_based(doc)
    content = doc.page_content.strip()
    preview = textwrap.shorten(content, width=600, placeholder=" ...")
    return RetrievedChunk(content=preview, page=page, score=score, metadata=metadata)


__all__ = ["build_vector_store", "retrieve_chunks", "chunk_from_document"]
