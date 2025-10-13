from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LangchainRAGConfig:
    """Configuration parameters controlling the LangChain RAG pipeline."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    embedding_dimension: int = 768
    restrict_to_task_pages: bool = True
    page_padding: int = 0
    use_mmr: bool = False
    mmr_lambda: float = 0.5
    mmr_fetch_k: int = 12
    max_context_characters: Optional[int] = 4000
    context_separator: str = "\n\n---\n\n"

    def describe(self) -> str:
        variant = "mmr" if self.use_mmr else "sim"
        return (
            f"chunk{self.chunk_size}-overlap{self.chunk_overlap}-"
            f"k{self.top_k}-{variant}-dim{self.embedding_dimension}"
        )


__all__ = ["LangchainRAGConfig"]
