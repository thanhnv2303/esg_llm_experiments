from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class LangchainRAGConfig:
    """Configuration parameters controlling the LangChain RAG pipeline."""

    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 10
    embedding_dimension: int = 768
    embedding_backend: str = "hash"
    embedding_model: Optional[str] = None
    embedding_kwargs: Dict[str, object] = field(default_factory=dict)
    restrict_to_task_pages: bool = True
    page_padding: int = 0
    use_mmr: bool = False
    mmr_lambda: float = 0.5
    mmr_fetch_k: int = 12
    max_context_characters: Optional[int] = 4000
    context_separator: str = "\n\n---\n\n"
    caption_charts: bool = True
    chart_caption_prompt: str = (
        "Write a concise, factual caption (one or two sentences) that quotes the key numbers and units exactly as shown in the chart."
    )
    chart_to_table_prompt: str = (
        "You analyse sustainability disclosures.\n"
        "You will receive a single chart or graph extracted from a corporate report.\n"
        "Respond with a JSON object containing the keys: is_chart (bool), caption (string or null), table_markdown (string or null), notes (string or null).\n"
        "If the image is not a quantitative chart, set is_chart=false and leave the other fields null.\n"
        "If it is a chart, set is_chart=true and produce BOTH a caption and a Markdown table in the same response.\n"
        "The caption must summarise the main quantitative insight in ≤2 sentences, quoting values and units exactly as shown.\n"
        "The Markdown table must include a header row and the numeric data from the chart (copy percentages/units verbatim).\n"
        "Use the notes field for any assumptions or data quality warnings."
    )
    chart_caption_max_images: Optional[int] = -1
    chart_caption_use_docling: bool = True
    extract_tables: bool = True
    table_use_docling: bool = True
    table_row_chunk_size: int = 50
    save_chunks_to_temp: bool = False
    chunk_dump_dir: Optional[Path] = None
    parse_cache_dir: Optional[Path] = Path(".rag_cache")
    debug_store_pages: bool = False
    debug_store_tables: bool = True
    debug_store_chart_tables: bool = True
    debug_store_captions: bool = False
    debug_store_chunks: bool = True
    debug_store_images: bool = True

    def describe(self) -> str:
        variant = "mmr" if self.use_mmr else "sim"
        backend = self.embedding_backend.replace("/", "-")
        if backend == "hash":
            embed_descriptor = f"{backend}-dim{self.embedding_dimension}"
        elif self.embedding_model:
            model_stub = self.embedding_model.split("/")[-1]
            safe_stub = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model_stub)
            embed_descriptor = f"{backend}-{safe_stub}"
        else:
            embed_descriptor = backend
        tables = "tbl" if self.extract_tables else "notbl"
        return (
            f"chunk{self.chunk_size}-overlap{self.chunk_overlap}-"
            f"k{self.top_k}-{variant}-{embed_descriptor}-{tables}"
        )


__all__ = ["LangchainRAGConfig"]
