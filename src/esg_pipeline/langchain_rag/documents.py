from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from ..config import TaskConfig
from .config import LangchainRAGConfig


class DocumentLoader:
    """Lazy PDF loader with optional page filtering based on task definitions."""

    def __init__(self) -> None:
        self._raw_documents: Optional[List[Document]] = None
        self._raw_document_source: Optional[Path] = None

    def load(self, document_path: Path, tasks: Sequence[TaskConfig], config: LangchainRAGConfig) -> List[Document]:
        documents = self._load_all(document_path)
        if not config.restrict_to_task_pages:
            return list(documents)
        requested_pages = expand_task_pages(tasks, config.page_padding)
        return [doc for doc in documents if page_one_based(doc) in requested_pages]

    def _load_all(self, document_path: Path) -> List[Document]:
        absolute_path = document_path.expanduser().resolve()
        if self._raw_documents is not None and self._raw_document_source == absolute_path:
            return self._raw_documents
        loader = PyPDFLoader(str(absolute_path))
        documents = loader.load()
        self._raw_documents = documents
        self._raw_document_source = absolute_path
        return documents


def expand_task_pages(tasks: Sequence[TaskConfig], padding: int) -> Iterable[int]:
    pages: set[int] = set()
    for task in tasks:
        base = max(1, task.page)
        for delta in range(-padding, padding + 1):
            pages.add(max(1, base + delta))
    return sorted(pages)


def page_one_based(doc: Document) -> Optional[int]:
    page = doc.metadata.get("page")
    if isinstance(page, int):
        return page + 1
    if isinstance(page, str) and page.isdigit():
        return int(page) + 1
    return None


__all__ = ["DocumentLoader", "expand_task_pages", "page_one_based"]
