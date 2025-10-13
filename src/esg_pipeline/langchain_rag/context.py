from __future__ import annotations

from typing import Iterable, List

from ..config import TaskConfig
from .config import LangchainRAGConfig
from .types import RetrievedChunk


def build_query(company: str, task: TaskConfig) -> str:
    terms: List[str] = [company, task.indicator]
    terms.extend(task.prompt_overrides.indicator_synonyms)
    if task.expected:
        terms.append(task.expected)
    return " ".join(term.strip() for term in terms if term and term.strip())


def format_context(chunks: Iterable[RetrievedChunk], config: LangchainRAGConfig) -> str:
    material = list(chunks)
    if not material:
        return "No relevant context retrieved."

    sections: List[str] = []
    for index, chunk in enumerate(material, start=1):
        header_bits = [f"Chunk {index}"]
        if chunk.page is not None:
            header_bits.append(f"page {chunk.page}")
        if chunk.score is not None:
            header_bits.append(f"score {chunk.score:.4f}")
        header = " | ".join(header_bits)
        sections.append(f"## {header}\n{chunk.content.strip()}")

    context = config.context_separator.join(sections)
    if config.max_context_characters and len(context) > config.max_context_characters:
        context = context[: config.max_context_characters] + "\n\n[truncated]"
    return context


def compose_prompt(base_prompt: str, context: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "Context provided by retrieval:\n"
        f"{context}\n\n"
        "Answer strictly using only this context."
    )


__all__ = ["build_query", "format_context", "compose_prompt"]
