from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..pipeline import TaskRunResult
from .types import RetrievedChunk


@dataclass
class LangchainRAGTaskResult(TaskRunResult):
    """Task outcome enriched with retrieval metadata."""

    query: str = ""
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    chart_insights: List[Dict[str, object]] = field(default_factory=list)
    chart_summary_path: Optional[Path] = None


__all__ = ["LangchainRAGTaskResult"]
