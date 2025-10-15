from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..models.base import ModelRunner
from ..preprocessing import extract_page_as_image, extract_page_with_docling
from .config import LangchainRAGConfig


LOGGER = logging.getLogger(__name__)


@dataclass
class ChartInsight:
    """Structured representation of a single chart image."""

    chart_id: str
    description_lines: List[str] = field(default_factory=list)
    caption: Optional[str] = None
    key_numbers: List[str] = field(default_factory=list)
    image_path: Optional[Path] = None
    raw_response: Optional[str] = None
    doc_caption: Optional[str] = None

    def render_for_injection(self) -> str:
        header = f"[chart:{self.chart_id}]"
        lines: List[str] = [header]
        if self.caption:
            lines.append(f"Caption: {self.caption}")
        elif self.doc_caption:
            lines.append(f"Caption: {self.doc_caption}")
        if self.key_numbers:
            joined_numbers = ", ".join(self.key_numbers)
            lines.append(f"Key numbers: {joined_numbers}")
        if self.description_lines:
            lines.append("Insights:")
            for item in self.description_lines:
                prefix = "- " if not item.startswith(("-", "*", "•", "1")) else ""
                lines.append(f"{prefix}{item}")
        return "\n".join(lines)


@dataclass
class ChartSummary:
    """Captures chart-level insights for a single page."""

    page: int
    insights: List[ChartInsight] = field(default_factory=list)
    combined_image_path: Optional[Path] = None
    model_name: Optional[str] = None
    prompt: Optional[str] = None
    summary_path: Optional[Path] = None

    def has_content(self) -> bool:
        return any(insight.description_lines or insight.caption for insight in self.insights)


class ChartCaptioner:
    """Generate textual summaries for chart images using a multimodal model."""

    def __init__(
        self,
        pdf_path: Path,
        config: LangchainRAGConfig,
        charts_dir: Path,
        model: Optional[ModelRunner],
    ) -> None:
        self.pdf_path = pdf_path.expanduser().resolve()
        self.config = config
        self.model = model
        self.charts_dir = charts_dir
        self.logger = LOGGER.getChild("ChartCaptioner")
        self._raw_dir = charts_dir / "raw"
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[int, ChartSummary] = {}

    def generate(self, pages: Iterable[int]) -> Dict[int, ChartSummary]:
        if not self.config.caption_charts:
            return {}
        if self.model is None:
            self.logger.warning(
                "Chart captioning is enabled but no chart-capable model was provided; skipping."
            )
            return {}

        summaries: Dict[int, ChartSummary] = {}
        unique_pages = sorted({page for page in pages if page is not None})
        for page in unique_pages:
            summary = self._summarise_page(page)
            if summary and summary.has_content():
                summaries[page] = summary
        return summaries

    def _summarise_page(self, page: int) -> Optional[ChartSummary]:
        cached = self._cache.get(page)
        if cached is not None:
            return cached

        images, combined = self._collect_images(page)
        if not images:
            summary = ChartSummary(page=page)
            self._cache[page] = summary
            return summary

        limit = self.config.chart_caption_max_images
        if limit is not None and limit > 0:
            images = images[:limit]

        insights: List[ChartInsight] = []
        for idx, (image_path, doc_caption) in enumerate(images, start=1):
            insight = self._describe_image(page, image_path, idx, doc_caption)
            if insight:
                insights.append(insight)

        summary = ChartSummary(
            page=page,
            insights=insights,
            combined_image_path=combined if combined and combined.exists() else None,
            model_name=getattr(self.model, "name", self.model.__class__.__name__) if self.model else None,
            prompt=self.config.chart_caption_prompt,
        )
        self._cache[page] = summary
        if summary.has_content():
            self._write_page_summary(summary)
        return summary

    def _collect_images(self, page: int) -> Tuple[List[Tuple[Path, Optional[str]]], Optional[Path]]:
        page_root = self._raw_dir / f"page_{page:04d}"
        images_dir = page_root / "images"
        markdown_dir = page_root / "markdown"
        images_dir.mkdir(parents=True, exist_ok=True)
        markdown_dir.mkdir(parents=True, exist_ok=True)

        captions: Dict[str, str] = {}
        if self.config.chart_caption_use_docling:
            try:
                artifacts = extract_page_with_docling(
                    self.pdf_path,
                    page,
                    markdown_dir=markdown_dir,
                    images_dir=images_dir,
                    image_mode="referenced",
                    filename_prefix=f"page{page:04d}",
                )
                image_paths = [path for path in artifacts.image_paths if path.exists()]
                combined_path = artifacts.combined_image_path
                captions = _map_docling_captions(
                    artifacts.markdown_text,
                    image_paths,
                )
                if image_paths:
                    return [(path, captions.get(path.name)) for path in image_paths], combined_path
            except Exception as exc:  # pragma: no cover - optional dependency path
                self.logger.warning(
                    "Docling extraction failed for page %s: %s", page, exc
                )

        try:
            fallback = extract_page_as_image(
                self.pdf_path,
                page,
                images_dir,
                filename_prefix=f"page{page:04d}",
            )
            return [(fallback, None)], None
        except Exception as exc:  # pragma: no cover - depends on optional tools
            self.logger.warning(
                "Failed to rasterise page %s for chart captioning: %s", page, exc
            )
            return [], None

    def _describe_image(
        self,
        page: int,
        image_path: Path,
        index: int,
        doc_caption: Optional[str],
    ) -> Optional[ChartInsight]:
        if self.model is None:
            return None

        prompt = (
            f"{self.config.chart_caption_prompt}\n\n"
            f"Image page: {page}. Asset name: {image_path.name}."
        )
        if doc_caption:
            prompt += f"\nDocument caption: {doc_caption}"
        try:
            response = self.model.predict(
                prompt,
                page_images=[image_path],
            )
        except Exception as exc:  # pragma: no cover - depends on external model
            self.logger.warning(
                "Chart captioning model failed for page %s image %s: %s",
                page,
                image_path,
                exc,
            )
            return None

        text = response.raw_response.strip()
        if not text:
            return None

        caption, description_lines = _parse_caption(text)
        key_numbers = _extract_numbers("\n".join(description_lines + ([caption] if caption else [])))
        if doc_caption and not caption:
            caption = doc_caption

        if not description_lines and not caption:
            return None

        chart_id = f"page{page:04d}_chart{index:02d}"
        insight = ChartInsight(
            chart_id=chart_id,
            description_lines=description_lines,
            caption=caption,
            key_numbers=key_numbers,
            image_path=image_path,
            raw_response=text,
            doc_caption=doc_caption,
        )
        return insight

    def _write_page_summary(self, summary: ChartSummary) -> None:
        page_summary_path = self.charts_dir / f"page_{summary.page:04d}.md"
        header = [f"# Chart insights – page {summary.page}"]
        if summary.model_name:
            header.append(f"_Model: {summary.model_name}_")
        body = []
        for insight in summary.insights:
            body.append(
                "\n".join(
                    [f"## {insight.chart_id}", insight.render_for_injection()]
                )
            )
        content = "\n\n".join(header + body)
        page_summary_path.write_text(content, encoding="utf-8")
        summary.summary_path = page_summary_path


def _parse_caption(text: str) -> Tuple[Optional[str], List[str]]:
    stripped_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not stripped_lines:
        return None, []

    json_candidate = _extract_json_payload("\n".join(stripped_lines))
    if json_candidate:
        caption = json_candidate.get("caption")
        insights = json_candidate.get("insights") or json_candidate.get("summary") or []
        numbers = json_candidate.get("key_numbers")
        lines = [line.strip() for line in insights if isinstance(line, str) and line.strip()]
        if caption and not isinstance(caption, str):
            caption = str(caption)
        if numbers and isinstance(numbers, list):
            # fold numbers into description lines for downstream extraction
            number_line = ", ".join(str(item) for item in numbers if item)
            if number_line:
                lines.append(f"Key numbers: {number_line}")
        if caption or lines:
            return caption, _normalise_bullets(lines)

    first_line = stripped_lines[0]
    caption = None
    remaining = stripped_lines
    if not first_line.startswith(("-", "*", "•", "1")):
        caption = first_line
        remaining = stripped_lines[1:]

    return caption, _normalise_bullets(remaining)


def _normalise_bullets(lines: List[str]) -> List[str]:
    normalised: List[str] = []
    for raw_line in lines:
        if not raw_line:
            continue
        prefix = raw_line[0]
        if prefix in {"-", "*", "•"} or (prefix.isdigit() and len(raw_line) > 1 and raw_line[1] in {".", ")"}):
            normalised.append(raw_line)
        else:
            normalised.append(f"- {raw_line}")
    return normalised


def _extract_numbers(text: str) -> List[str]:
    if not text:
        return []
    matches = re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?", text)
    seen: Dict[str, None] = {}
    for match in matches:
        key = match.strip().rstrip(",")
        if key and key not in seen:
            seen[key] = None
    return list(seen.keys())


def _extract_json_payload(text: str) -> Dict[str, object]:
    candidates: List[str] = []
    if not text:
        return {}
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    fence_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        candidates.append(match.group(1))
    brace_pattern = re.compile(r"\{[^{}]+\}")
    for snippet in brace_pattern.findall(text):
        candidates.append(snippet)

    for snippet in candidates:
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return {}


def _map_docling_captions(markdown_text: str, image_paths: List[Path]) -> Dict[str, str]:
    if not markdown_text:
        return {}
    pattern = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
    captions: Dict[str, str] = {}
    for match in pattern.finditer(markdown_text):
        alt = match.group("alt").strip()
        rel_path = Path(match.group("path")).name
        if alt:
            captions[rel_path] = alt
    # Ensure only returning entries for existing paths
    result: Dict[str, str] = {}
    for path in image_paths:
        name = path.name
        if name in captions:
            result[name] = captions[name]
    return result


__all__ = ["ChartCaptioner", "ChartSummary", "ChartInsight"]
