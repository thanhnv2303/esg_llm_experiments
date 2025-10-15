from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from langchain_core.documents import Document

from .config import LangchainRAGConfig

try:  # pragma: no cover - optional dependency
    from ..preprocessing.docling_backend import extract_page_with_docling
except ImportError:  # pragma: no cover - optional dependency
    extract_page_with_docling = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


@dataclass
class TableBlock:
    lines: List[str]
    caption: Optional[str]
    context_above: Optional[str]
    context_below: Optional[str]


class TableExtractor:
    """Extracts tabular data from PDF pages and prepares retrieval-ready chunks."""

    def __init__(self, pdf_path: Path) -> None:
        self.pdf_path = pdf_path.expanduser().resolve()
        self._page_table_counts: dict[int, int] = {}

    def extract(self, pages: Iterable[int], config: LangchainRAGConfig) -> List[Document]:
        if not config.extract_tables:
            return []
        if not pages:
            return []
        if not config.table_use_docling:
            LOGGER.warning(
                "Table extraction requested but 'table_use_docling' disabled; skipping for %s",
                self.pdf_path,
            )
            return []
        if extract_page_with_docling is None:  # pragma: no cover - optional dependency
            LOGGER.warning(
                "Docling not available; unable to extract tables from %s",
                self.pdf_path,
            )
            return []

        docs: List[Document] = []
        unique_pages = sorted({page for page in pages if page is not None and page > 0})
        for page in unique_pages:
            try:
                docs.extend(self._extract_page(page, config))
            except Exception as exc:  # pragma: no cover - depends on optional dependency
                LOGGER.warning("Failed to extract tables for page %s: %s", page, exc)
        return docs

    def _extract_page(self, page: int, config: LangchainRAGConfig) -> List[Document]:
        blocks = self._parse_page_tables(page)
        if not blocks:
            return []
        return self.documents_from_blocks(page=page, blocks=blocks, config=config)

    def documents_from_blocks(
        self,
        *,
        page: int,
        blocks: Sequence[TableBlock],
        config: LangchainRAGConfig,
    ) -> List[Document]:
        documents: List[Document] = []
        for block in blocks:
            table_index = self._page_table_counts.get(page, 0) + 1
            self._page_table_counts[page] = table_index
            table_id = f"page{page:04d}_table{table_index:02d}"
            documents.extend(
                self._create_table_documents(
                    table_id=table_id,
                    page=page,
                    block=block,
                    config=config,
                )
            )
        return documents

    def _parse_page_tables(self, page: int) -> List[TableBlock]:
        assert extract_page_with_docling is not None
        with tempfile.TemporaryDirectory(prefix=f"docling_tables_{page:04d}_") as tmp:
            tmp_path = Path(tmp)
            markdown_dir = tmp_path / "markdown"
            markdown_dir.mkdir(parents=True, exist_ok=True)
            artifacts = extract_page_with_docling(
                self.pdf_path,
                page,
                markdown_dir=markdown_dir,
                images_dir=None,
                image_mode="embedded",
                filename_prefix=f"page{page:04d}_table",
            )
            markdown_text = artifacts.markdown_text

        if not markdown_text:
            return []

        return parse_tables_from_markdown(markdown_text)

    def _create_table_documents(
        self,
        *,
        table_id: str,
        page: int,
        block: TableBlock,
        config: LangchainRAGConfig,
    ) -> List[Document]:
        header, separator, rows = _split_table_lines(block.lines)
        if not header or not separator:
            return []

        rows_per_chunk = max(1, config.table_row_chunk_size)
        row_chunks: List[Sequence[str]]
        if len(rows) <= rows_per_chunk:
            row_chunks = [rows]
        else:
            row_chunks = [rows[i : i + rows_per_chunk] for i in range(0, len(rows), rows_per_chunk)]

        title_candidates = [block.caption or "", block.context_above or ""]
        title = next((candidate.strip() for candidate in title_candidates if candidate and candidate.strip()), None)
        context_lines = [line for line in [block.context_above, block.context_below] if line]
        context_text = " \n".join(context_lines) if context_lines else None

        documents: List[Document] = []
        total_parts = len(row_chunks)
        full_markdown = "\n".join([header, separator, *rows])
        for index, chunk_rows in enumerate(row_chunks, start=1):
            chunk_lines = [header, separator, *chunk_rows]
            markdown = "\n".join(chunk_lines)
            parts: List[str] = [f"[table:{table_id}]" if total_parts == 1 else f"[table:{table_id} part {index}/{total_parts}]"]
            if title:
                parts.append(f"Title: {title}")
            elif block.caption:
                parts.append(f"Caption: {block.caption}")
            if context_text:
                parts.append(f"Context: {context_text}")
            parts.append(markdown)
            content = "\n\n".join(parts)

            metadata = {
                "source": str(self.pdf_path),
                "page": page - 1,
                "type": "table",
                "table_id": table_id,
                "table_caption": block.caption,
                "table_title": title,
                "table_context": context_text,
                "table_chunk_index": index,
                "table_chunk_count": total_parts,
                "rag_pre_chunked": True,
                "table_header": _split_cells(header),
                "table_full_markdown": full_markdown,
            }
            chunk_doc = Document(page_content=content, metadata=metadata)
            documents.append(chunk_doc)
        return documents


def parse_tables_from_markdown(markdown_text: str) -> List[TableBlock]:
    if not markdown_text:
        return []

    lines = markdown_text.splitlines()
    blocks: List[TableBlock] = []
    i = 0
    while i < len(lines):
        if _is_table_header(lines, i):
            start = i
            i += 2  # skip header + separator
            while i < len(lines) and _is_table_row(lines[i]):
                i += 1
            table_lines = lines[start:i]
            caption, context_above, context_below = _derive_context(lines, start, i)
            blocks.append(
                TableBlock(
                    lines=[line.rstrip() for line in table_lines if line.strip()],
                    caption=caption,
                    context_above=context_above,
                    context_below=context_below,
                )
            )
        else:
            i += 1
    return blocks


def _is_table_header(lines: List[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    header = lines[index].strip()
    separator = lines[index + 1].strip()
    if not _is_table_row(header) or not _is_separator(separator):
        return False
    return True


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and "|" in stripped[1:]


def _is_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    return bool(re.match(r"\|[\s:-]+\|", stripped))


def _derive_context(lines: List[str], start: int, end: int) -> tuple[Optional[str], Optional[str], Optional[str]]:
    caption = None
    context_above = None
    context_below = None

    for idx in range(start - 1, -1, -1):
        candidate = lines[idx].strip()
        if not candidate:
            continue
        context_above = candidate
        if "table" in candidate.lower() or candidate.endswith(":"):
            caption = candidate.rstrip(":")
        break

    for idx in range(end, len(lines)):
        candidate = lines[idx].strip()
        if not candidate:
            continue
        context_below = candidate
        break

    return caption, context_above, context_below


def _split_table_lines(lines: List[str]) -> tuple[Optional[str], Optional[str], List[str]]:
    if len(lines) < 2:
        return None, None, []
    header = lines[0]
    separator = lines[1]
    rows = lines[2:]
    return header, separator, rows


def _split_cells(line: str) -> List[str]:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return [cell for cell in cells if cell]


__all__ = ["TableExtractor", "TableBlock", "parse_tables_from_markdown"]
