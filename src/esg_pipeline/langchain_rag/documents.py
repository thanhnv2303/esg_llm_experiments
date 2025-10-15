from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from ..config import TaskConfig
from ..preprocessing import extract_page_with_docling
from .config import LangchainRAGConfig
from .tables import TableBlock, TableExtractor, parse_tables_from_markdown

try:  # pragma: no cover - optional dependency
    import pytesseract  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from PIL import Image  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..models.base import ModelRunner


LOGGER = logging.getLogger(__name__)


@dataclass
class PageSnapshot:
    page: int
    text: str
    markdown: str
    image_paths: List[Path]
    cache_dir: Optional[Path]


@dataclass
class ChartCandidate:
    chart_id: str
    page: int
    image_path: Path
    relative_path: Path
    doc_caption: Optional[str]
    alt_text: Optional[str]
    context_above: Optional[str]
    context_below: Optional[str]
    is_chart_like: bool
    ocr_text: Optional[str] = None


@dataclass
class ChartConversion:
    chart_id: str
    page: int
    caption: Optional[str]
    doc_caption: Optional[str]
    table_markdown: Optional[str]
    notes: Optional[str]
    raw_response: Optional[str]
    image_path: Optional[Path]
    table_id: Optional[str] = None
    table_document: Optional[Document] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    table_header: Optional[List[str]] = None
    ocr_text: Optional[str] = None


@dataclass
class PageData:
    page: int
    snapshot: PageSnapshot
    table_blocks: List[TableBlock] = field(default_factory=list)
    chart_candidates: List[ChartCandidate] = field(default_factory=list)


@dataclass
class PreparedDocument:
    documents: List[Document]
    text_documents: List[Document]
    table_documents: List[Document]
    page_markdowns: Dict[int, str]
    charts_by_page: Dict[int, List[ChartConversion]]
    chart_inventory: Dict[str, Dict[str, object]]
    snapshots: Dict[int, PageSnapshot]


class DocumentLoader:
    """Parse PDF pages into Markdown/text blocks, tables, and chart-derived tables."""

    def __init__(self) -> None:
        self._cached_documents: Optional[List[Document]] = None
        self._cached_source: Optional[Path] = None
        self._transient_dirs: List[tempfile.TemporaryDirectory[str]] = []
        self._docling_warning_emitted = False
        self._ocr_cache: Dict[Path, Optional[str]] = {}
        self._ocr_warning_emitted = False

    def prepare(
        self,
        document_path: Path,
        tasks: Sequence[TaskConfig],
        config: LangchainRAGConfig,
        chart_model: Optional["ModelRunner"] = None,
    ) -> PreparedDocument:
        pdf_path = document_path.expanduser().resolve()
        pdf_documents = self._load_pdf(pdf_path)
        page_map: Dict[int, Document] = {
            page_one_based(doc): doc for doc in pdf_documents if page_one_based(doc) is not None
        }

        requested_pages = self._determine_pages(pdf_documents, tasks, config)
        cache_base = self._get_cache_base_dir(pdf_path, config)

        page_data: List[PageData] = []
        for page in requested_pages:
            base_doc = page_map.get(page)
            base_text = base_doc.page_content if base_doc else ""
            snapshot = self._load_page_snapshot(pdf_path, page, base_text, config, cache_base)
            table_blocks: List[TableBlock] = []
            if snapshot.markdown and config.extract_tables and config.table_use_docling:
                table_blocks = parse_tables_from_markdown(snapshot.markdown)
            candidates: List[ChartCandidate] = []
            if config.caption_charts:
                candidates = self._build_chart_candidates(snapshot)
            page_data.append(PageData(page=page, snapshot=snapshot, table_blocks=table_blocks, chart_candidates=candidates))

        table_extractor = TableExtractor(pdf_path)
        table_documents: List[Document] = []
        table_docs_by_page: Dict[int, List[Document]] = {page.page: [] for page in page_data}
        table_refs_by_page: Dict[int, List[Dict[str, object]]] = {page.page: [] for page in page_data}

        for data in page_data:
            if not data.table_blocks:
                continue
            docs = table_extractor.documents_from_blocks(
                page=data.page,
                blocks=data.table_blocks,
                config=config,
            )
            if not docs:
                continue
            table_documents.extend(docs)
            table_docs_by_page[data.page].extend(docs)
            table_refs_by_page[data.page] = self._collect_table_references(docs)

        chart_conversions_by_page: Dict[int, List[ChartConversion]] = {data.page: [] for data in page_data}
        chart_inventory: Dict[str, Dict[str, object]] = {}

        if config.caption_charts and chart_model is not None:
            for data in page_data:
                if not data.chart_candidates:
                    continue
                limit = config.chart_caption_max_images
                chart_candidates = [candidate for candidate in data.chart_candidates if candidate.is_chart_like]
                if limit is not None and limit > 0:
                    chart_candidates = chart_candidates[:limit]
                conversions: List[ChartConversion] = []
                for candidate in chart_candidates:
                    conversion = self._convert_chart_candidate(candidate, chart_model, config)
                    conversions.append(conversion)
                    if conversion.table_document is not None:
                        table_documents.append(conversion.table_document)
                        table_docs_by_page[data.page].append(conversion.table_document)
                        table_refs_by_page[data.page].append(
                            self._chart_table_reference(conversion)
                        )
                        chart_inventory[conversion.chart_id] = self._build_chart_inventory_record(conversion)
                chart_conversions_by_page[data.page] = conversions
        else:
            # Record skipped conversions when chart detection is enabled but no model provided.
            for data in page_data:
                candidates = [candidate for candidate in data.chart_candidates if candidate.is_chart_like]
                if not candidates:
                    continue
                conversions: List[ChartConversion] = []
                for candidate in candidates:
                    conversions.append(
                        ChartConversion(
                            chart_id=candidate.chart_id,
                            page=candidate.page,
                            caption=None,
                            doc_caption=candidate.doc_caption,
                            table_markdown=None,
                            notes=None,
                            raw_response=None,
                            image_path=candidate.image_path,
                            skipped=True,
                            skip_reason="Chart model not available",
                            ocr_text=candidate.ocr_text,
                        )
                    )
                chart_conversions_by_page[data.page] = conversions

        text_documents: List[Document] = []
        combined_documents: List[Document] = []
        page_markdowns: Dict[int, str] = {}

        for data in page_data:
            table_refs = table_refs_by_page.get(data.page, [])
            conversions = chart_conversions_by_page.get(data.page, [])
            text_doc = self._build_page_text_document(pdf_path, data, table_refs, conversions)
            text_documents.append(text_doc)
            combined_documents.append(text_doc)
            combined_documents.extend(table_docs_by_page.get(data.page, []))

            if (
                config.debug_store_pages
                or config.debug_store_tables
                or config.debug_store_chart_tables
            ):
                page_markdowns[data.page] = self._build_page_markdown(data, table_docs_by_page.get(data.page, []), conversions)

        return PreparedDocument(
            documents=combined_documents,
            text_documents=text_documents,
            table_documents=table_documents,
            page_markdowns=page_markdowns,
            charts_by_page=chart_conversions_by_page,
            chart_inventory=chart_inventory,
            snapshots={data.page: data.snapshot for data in page_data},
        )

    def _load_pdf(self, pdf_path: Path) -> List[Document]:
        if self._cached_documents is not None and self._cached_source == pdf_path:
            return self._cached_documents
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        self._cached_documents = documents
        self._cached_source = pdf_path
        return documents

    def _determine_pages(
        self,
        documents: Sequence[Document],
        tasks: Sequence[TaskConfig],
        config: LangchainRAGConfig,
    ) -> List[int]:
        if not config.restrict_to_task_pages:
            pages = [page for page in (page_one_based(doc) for doc in documents) if page is not None]
            return sorted(dict.fromkeys(pages))
        return list(expand_task_pages(tasks, config.page_padding))

    def _get_cache_base_dir(
        self,
        pdf_path: Path,
        config: LangchainRAGConfig,
    ) -> Optional[Path]:
        if not config.parse_cache_dir:
            return None
        stat = pdf_path.stat()
        signature = hashlib.sha1(
            f"{pdf_path.resolve()}::{stat.st_size}::{int(stat.st_mtime_ns)}".encode("utf-8")
        ).hexdigest()
        base_dir = config.parse_cache_dir.expanduser().resolve()
        doc_dir = base_dir / signature
        doc_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = doc_dir / "metadata.json"
        if not metadata_path.exists():
            metadata_path.write_text(
                json.dumps(
                    {
                        "source": str(pdf_path.resolve()),
                        "size": stat.st_size,
                        "mtime_ns": int(stat.st_mtime_ns),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return doc_dir

    def _load_page_snapshot(
        self,
        pdf_path: Path,
        page: int,
        base_text: str,
        config: LangchainRAGConfig,
        cache_base: Optional[Path],
    ) -> PageSnapshot:
        docling_required = (
            (config.extract_tables and config.table_use_docling)
            or (config.caption_charts and config.chart_caption_use_docling)
        )
        if not docling_required:
            return PageSnapshot(page=page, text=base_text, markdown="", image_paths=[], cache_dir=None)

        if cache_base is not None:
            page_dir = cache_base / f"page_{page:04d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            cache_file = page_dir / "page.json"
            if cache_file.exists():
                try:
                    with cache_file.open("r", encoding="utf-8") as fh:
                        payload = json.load(fh)
                    markdown = payload.get("markdown", "")
                    text_override = payload.get("text") or base_text
                    image_rel_paths = [page_dir / rel for rel in payload.get("images", [])]
                    image_paths = [path for path in image_rel_paths if path.exists()]
                    ocr_map = {
                        (page_dir / rel): payload.get("ocr", {}).get(rel)
                        for rel in payload.get("images", [])
                    }
                    snapshot = PageSnapshot(
                        page=page,
                        text=text_override,
                        markdown=markdown,
                        image_paths=image_paths,
                        cache_dir=page_dir,
                    )
                    for image_path in image_paths:
                        if image_path in ocr_map and ocr_map[image_path] is not None:
                            self._ocr_cache[image_path] = ocr_map[image_path]
                    return snapshot
                except json.JSONDecodeError:
                    pass  # fall back to re-processing
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix=f"rag_parse_{page:04d}_")
            self._transient_dirs.append(temp_dir)
            page_dir = Path(temp_dir.name)

        markdown_text = ""
        image_paths: List[Path] = []
        try:
            artifacts = extract_page_with_docling(
                pdf_path,
                page,
                markdown_dir=page_dir,
                images_dir=page_dir / "images",
                image_mode="referenced",
                filename_prefix=f"page{page:04d}",
            )
            markdown_text = artifacts.markdown_text or ""
            image_paths = [path for path in artifacts.image_paths if path.exists()]
        except Exception as exc:  # pragma: no cover - optional dependency path
            if not self._docling_warning_emitted:
                LOGGER.warning("Docling extraction failed for %s page %s: %s", pdf_path, page, exc)
                self._docling_warning_emitted = True
            markdown_text = ""
            image_paths = []

        if cache_base is not None:
            images_rel = []
            ocr_payload: Dict[str, Optional[str]] = {}
            for path in image_paths:
                try:
                    rel = str(path.relative_to(page_dir))
                except ValueError:
                    rel = path.name
                images_rel.append(rel)
                ocr_payload[rel] = self._ocr_cache.get(path)
            cache_payload = {
                "page": page,
                "text": base_text,
                "markdown": markdown_text,
                "images": images_rel,
                "ocr": ocr_payload,
            }
            cache_file = page_dir / "page.json"
            cache_file.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")

        return PageSnapshot(
            page=page,
            text=base_text,
            markdown=markdown_text,
            image_paths=image_paths,
            cache_dir=page_dir,
        )

    def _build_chart_candidates(
        self,
        snapshot: PageSnapshot,
    ) -> List[ChartCandidate]:
        if not snapshot.image_paths:
            return []

        entries = self._extract_image_entries(snapshot.markdown)
        candidates: List[ChartCandidate] = []
        for index, image_path in enumerate(sorted(snapshot.image_paths)):
            record = entries.get(image_path.name, {})
            doc_caption = record.get("alt") or record.get("context_above") or record.get("context_below")
            alt_text = record.get("alt")
            context_above = record.get("context_above")
            context_below = record.get("context_below")
            ocr_text = self._extract_ocr_text(image_path)
            if ocr_text is None:
                is_chart_like = True
            else:
                has_letters = bool(re.search(r"[A-Za-z]", ocr_text))
                has_numbers = bool(re.search(r"\d", ocr_text))
                is_chart_like = has_letters or has_numbers
            if snapshot.cache_dir is not None:
                try:
                    relative_path = image_path.relative_to(snapshot.cache_dir)
                except ValueError:
                    relative_path = Path(image_path.name)
            else:
                relative_path = Path(image_path.name)
            candidates.append(
                ChartCandidate(
                    chart_id=f"page{snapshot.page:04d}_chart{index + 1:02d}",
                    page=snapshot.page,
                    image_path=image_path,
                    relative_path=relative_path,
                    doc_caption=doc_caption,
                    alt_text=alt_text,
                    context_above=context_above,
                    context_below=context_below,
                    is_chart_like=is_chart_like,
                    ocr_text=ocr_text,
                )
            )
        return candidates

    def _collect_table_references(self, docs: Sequence[Document]) -> List[Dict[str, object]]:
        references: Dict[str, Dict[str, object]] = {}
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            if metadata.get("type") != "table":
                continue
            table_id = metadata.get("table_id")
            chunk_index = metadata.get("table_chunk_index")
            if not table_id:
                continue
            record = references.setdefault(
                table_id,
                {
                    "table_id": table_id,
                    "caption": metadata.get("table_caption"),
                    "title": metadata.get("table_title") or metadata.get("table_caption"),
                    "origin": metadata.get("table_origin", "document"),
                },
            )
            if chunk_index == 1 or chunk_index is None:
                record["caption"] = metadata.get("table_caption")
                record["title"] = metadata.get("table_title") or metadata.get("table_caption")
        return list(references.values())

    def _convert_chart_candidate(
        self,
        candidate: ChartCandidate,
        chart_model: "ModelRunner",
        config: LangchainRAGConfig,
    ) -> ChartConversion:
        if not candidate.is_chart_like:
            return ChartConversion(
                chart_id=candidate.chart_id,
                page=candidate.page,
                caption=None,
                doc_caption=candidate.doc_caption,
                table_markdown=None,
                notes=None,
                raw_response=None,
                image_path=candidate.image_path,
                skipped=True,
                skip_reason="OCR detected no text or numbers",
                ocr_text=candidate.ocr_text,
            )

        prompt_sections = [config.chart_to_table_prompt.strip()]
        caption_context = candidate.doc_caption or candidate.alt_text or "(no caption provided)"
        prompt_sections.append(f"Document-caption: {caption_context}")
        if config.chart_caption_prompt:
            prompt_sections.append("Caption requirements: " + config.chart_caption_prompt.strip())
        prompt_sections.append("Return JSON only.")
        prompt = "\n\n".join(section for section in prompt_sections if section)

        try:
            response = chart_model.predict(prompt, page_images=[candidate.image_path])
        except Exception as exc:  # pragma: no cover - depends on external backends
            return ChartConversion(
                chart_id=candidate.chart_id,
                page=candidate.page,
                caption=None,
                doc_caption=candidate.doc_caption,
                table_markdown=None,
                notes=None,
                raw_response=None,
                image_path=candidate.image_path,
                skipped=True,
                skip_reason=f"Chart model error: {exc}",
                ocr_text=candidate.ocr_text,
            )

        raw_text = response.raw_response.strip()
        payload = self._extract_json_payload(raw_text)
        if not payload:
            return ChartConversion(
                chart_id=candidate.chart_id,
                page=candidate.page,
                caption=None,
                doc_caption=candidate.doc_caption,
                table_markdown=None,
                notes=None,
                raw_response=raw_text,
                image_path=candidate.image_path,
                skipped=True,
                skip_reason="Model response lacked JSON payload",
                ocr_text=candidate.ocr_text,
            )

        is_chart = payload.get("is_chart")
        if isinstance(is_chart, str):
            is_chart = is_chart.lower().strip() in {"true", "yes", "1"}
        if is_chart is False:
            return ChartConversion(
                chart_id=candidate.chart_id,
                page=candidate.page,
                caption=None,
                doc_caption=candidate.doc_caption,
                table_markdown=None,
                notes=str(payload.get("notes")) if payload.get("notes") is not None else None,
                raw_response=raw_text,
                image_path=candidate.image_path,
                skipped=True,
                skip_reason="Model classified image as non-chart",
                ocr_text=candidate.ocr_text,
            )

        table_markdown_raw = payload.get("table_markdown") or payload.get("table") or payload.get("markdown")
        if isinstance(table_markdown_raw, list):
            table_markdown_raw = "\n".join(str(item) for item in table_markdown_raw)
        table_markdown = self._normalise_markdown_table(str(table_markdown_raw) if table_markdown_raw else "")
        if not table_markdown:
            return ChartConversion(
                chart_id=candidate.chart_id,
                page=candidate.page,
                caption=str(payload.get("caption")) if payload.get("caption") is not None else candidate.doc_caption,
                doc_caption=candidate.doc_caption,
                table_markdown=None,
                notes=str(payload.get("notes")) if payload.get("notes") is not None else None,
                raw_response=raw_text,
                image_path=candidate.image_path,
                skipped=True,
                skip_reason="Model did not return a Markdown table",
                ocr_text=candidate.ocr_text,
            )

        caption = payload.get("caption")
        if caption is not None and not isinstance(caption, str):
            caption = str(caption)
        if not caption:
            caption = candidate.doc_caption or candidate.alt_text

        notes = payload.get("notes")
        if notes is not None and not isinstance(notes, str):
            notes = str(notes)

        table_document, header = self._build_chart_table_document(
            candidate=candidate,
            caption=caption,
            table_markdown=table_markdown,
        )

        if table_document is None:
            return ChartConversion(
                chart_id=candidate.chart_id,
                page=candidate.page,
                caption=caption,
                doc_caption=candidate.doc_caption,
                table_markdown=None,
                notes=notes,
                raw_response=raw_text,
                image_path=candidate.image_path,
                skipped=True,
                skip_reason="Unable to interpret Markdown table",
            )

        return ChartConversion(
            chart_id=candidate.chart_id,
            page=candidate.page,
            caption=caption,
            doc_caption=candidate.doc_caption,
            table_markdown=table_markdown,
            notes=notes,
            raw_response=raw_text,
            image_path=candidate.image_path,
            table_document=table_document,
            table_id=table_document.metadata.get("table_id"),
            table_header=header,
            skipped=False,
            ocr_text=candidate.ocr_text,
        )

    def _build_chart_table_document(
        self,
        *,
        candidate: ChartCandidate,
        caption: Optional[str],
        table_markdown: str,
    ) -> Tuple[Optional[Document], Optional[List[str]]]:
        parsed = self._parse_markdown_table_lines(table_markdown)
        if parsed is None:
            return None, None
        header_line, separator_line, row_lines = parsed
        header_cells = self._split_cells(header_line)
        lines = [header_line, separator_line, *row_lines]
        content_lines = [f"[table:{candidate.chart_id}_table]", f"Caption: {caption or candidate.doc_caption or '-'}"]
        content_lines.append(f"Image path: {candidate.image_path}")
        content_lines.extend(lines)
        doc = Document(
            page_content="\n\n".join(content_lines),
            metadata={
                "source": str(candidate.image_path),
                "page": candidate.page - 1,
                "type": "table",
                "table_id": f"{candidate.chart_id}_table",
                "table_caption": caption,
                "table_title": caption,
                "table_context": candidate.doc_caption,
                "table_origin": "chart",
                "rag_pre_chunked": True,
                "table_chunk_index": 1,
                "table_chunk_count": 1,
                "chart_id": candidate.chart_id,
                "table_header": header_cells,
                "table_full_markdown": table_markdown,
            },
        )
        return doc, header_cells

    def _chart_table_reference(self, conversion: ChartConversion) -> Dict[str, object]:
        return {
            "table_id": conversion.table_id,
            "caption": conversion.caption,
            "title": conversion.caption or conversion.doc_caption,
            "origin": "chart",
        }

    def _build_chart_inventory_record(self, conversion: ChartConversion) -> Dict[str, object]:
        record: Dict[str, object] = {
            "chart_id": conversion.chart_id,
            "page": conversion.page,
            "caption": conversion.caption,
            "doc_caption": conversion.doc_caption,
            "table_id": conversion.table_id,
            "table_markdown": conversion.table_markdown,
            "notes": conversion.notes,
            "image_path": str(conversion.image_path) if conversion.image_path else None,
        }
        if conversion.table_header:
            record["table_header"] = conversion.table_header
        if conversion.ocr_text is not None:
            record["ocr_text"] = conversion.ocr_text
        return record

    def _build_page_text_document(
        self,
        pdf_path: Path,
        data: PageData,
        table_refs: Sequence[Dict[str, object]],
        conversions: Sequence[ChartConversion],
    ) -> Document:
        text = data.snapshot.text.strip()
        sections: List[str] = [text] if text else []

        if table_refs:
            table_lines = ["Tables referenced on this page:"]
            for ref in table_refs:
                table_id = ref.get("table_id")
                title = ref.get("title") or ref.get("caption") or "(untitled table)"
                origin = ref.get("origin", "document")
                table_lines.append(f"- [table:{table_id}] {title} ({origin})")
            sections.append("\n".join(table_lines))

        successful_conversions = [conv for conv in conversions if not conv.skipped and conv.table_id]
        if successful_conversions:
            chart_lines = ["Charts converted to tables:"]
            for conv in successful_conversions:
                caption = conv.caption or conv.doc_caption or "Chart"
                chart_lines.append(f"- [chart:{conv.chart_id}] {caption}")
            sections.append("\n".join(chart_lines))

        if not sections:
            sections.append("(No text extracted from this page.)")

        page_text = "\n\n".join(sections)
        return Document(
            page_content=page_text,
            metadata={
                "source": str(pdf_path),
                "page": data.page - 1,
                "type": "page_text",
            },
        )

    def _build_page_markdown(
        self,
        data: PageData,
        table_docs: Sequence[Document],
        conversions: Sequence[ChartConversion],
    ) -> str:
        sections = [f"# Page {data.page}"]
        text = data.snapshot.text.strip()
        sections.append(text or "(No text extracted)")

        if table_docs:
            sections.append("## Tables")
            for doc in table_docs:
                sections.append("---")
                sections.append(doc.page_content)

        if conversions:
            sections.append("## Charts")
            for conv in conversions:
                sections.append(f"### {conv.chart_id}")
                if conv.skipped:
                    sections.append(f"_Skipped: {conv.skip_reason}_")
                    continue
                if conv.caption:
                    sections.append(f"Caption: {conv.caption}")
                if conv.table_markdown:
                    sections.append(conv.table_markdown)
                if conv.notes:
                    sections.append(f"Notes: {conv.notes}")

        return "\n\n".join(sections)

    @staticmethod
    def _extract_image_entries(markdown_text: str) -> Dict[str, Dict[str, Optional[str]]]:
        entries: Dict[str, Dict[str, Optional[str]]] = {}
        if not markdown_text:
            return entries
        pattern = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
        lines = markdown_text.splitlines()
        for idx, line in enumerate(lines):
            match = pattern.search(line)
            if not match:
                continue
            rel = Path(match.group("path")).name
            alt = match.group("alt").strip() or None
            context_above = None
            context_below = None
            for j in range(idx - 1, -1, -1):
                candidate = lines[j].strip()
                if candidate:
                    context_above = candidate
                    break
            for j in range(idx + 1, len(lines)):
                candidate = lines[j].strip()
                if candidate:
                    context_below = candidate
                    break
            entries[rel] = {
                "alt": alt,
                "context_above": context_above,
                "context_below": context_below,
            }
        return entries

    @staticmethod
    def _extract_json_payload(text: str) -> Dict[str, object]:
        if not text:
            return {}
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\\s*(\{.*?\})\\s*```", text, flags=re.DOTALL)
        if fenced:
            snippet = fenced.group(1)
            try:
                data = json.loads(snippet)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if brace_match:
            snippet = brace_match.group(0)
            try:
                data = json.loads(snippet)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        return {}

    @staticmethod
    def _normalise_markdown_table(table_text: str) -> Optional[str]:
        if not table_text:
            return None
        fenced = re.search(r"```(?:markdown|md|table)?\\s*(.*?)```", table_text, flags=re.DOTALL)
        if fenced:
            table_text = fenced.group(1)
        lines = [line.rstrip() for line in table_text.splitlines() if line.strip()]
        if len(lines) < 2:
            return None
        table_lines: List[str] = []
        collecting = False
        for line in lines:
            if line.strip().startswith("|"):
                collecting = True
                table_lines.append(line.strip())
            elif collecting:
                break
        if len(table_lines) < 2:
            return None
        separator = table_lines[1]
        if "-" not in separator:
            return None
        return "\n".join(table_lines)

    @staticmethod
    def _parse_markdown_table_lines(table_markdown: str) -> Optional[Tuple[str, str, List[str]]]:
        lines = [line.strip() for line in table_markdown.splitlines() if line.strip()]
        if len(lines) < 2:
            return None
        header = lines[0]
        separator = lines[1]
        if not header.startswith("|") or not separator.startswith("|"):
            return None
        rows: List[str] = []
        for line in lines[2:]:
            if line.startswith("|"):
                rows.append(line)
            else:
                break
        if not rows:
            return None
        return header, separator, rows

    @staticmethod
    def _split_cells(line: str) -> List[str]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        return [cell for cell in cells if cell]

    def _extract_ocr_text(self, image_path: Path) -> Optional[str]:
        cached = self._ocr_cache.get(image_path)
        if cached is not None:
            return cached

        if pytesseract is None or Image is None:  # pragma: no cover - optional dependency
            if not self._ocr_warning_emitted:
                LOGGER.warning(
                    "pytesseract or Pillow not available; skipping OCR-based chart filtering."
                )
                self._ocr_warning_emitted = True
            self._ocr_cache[image_path] = None
            return None

        text = ""
        try:
            with Image.open(image_path) as img:
                text = pytesseract.image_to_string(img.convert("RGB"))
        except Exception as exc:  # pragma: no cover - depends on external tools
            LOGGER.debug("OCR failed for %s: %s", image_path, exc)
        cleaned = text.strip()
        self._ocr_cache[image_path] = cleaned
        return cleaned


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


__all__ = [
    "DocumentLoader",
    "PreparedDocument",
    "ChartCandidate",
    "ChartConversion",
    "expand_task_pages",
    "page_one_based",
]
