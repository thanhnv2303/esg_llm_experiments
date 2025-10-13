from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..pipeline import TaskRunResult
from .types import RetrievedChunk


@dataclass
class LangchainRAGTaskResult(TaskRunResult):
    """Task outcome enriched with retrieval metadata."""

    query: str = ""
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)


__all__ = ["LangchainRAGTaskResult"]
