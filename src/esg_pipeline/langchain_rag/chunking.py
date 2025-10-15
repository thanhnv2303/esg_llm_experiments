from __future__ import annotations

from typing import List, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import LangchainRAGConfig


def split_documents(documents: Sequence[Document], config: LangchainRAGConfig) -> Sequence[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True,
    )

    processed: List[Document] = []
    for doc in documents:
        if doc.metadata.get("rag_pre_chunked"):
            processed.append(doc)
        else:
            processed.extend(splitter.split_documents([doc]))
    return processed


__all__ = ["split_documents"]
