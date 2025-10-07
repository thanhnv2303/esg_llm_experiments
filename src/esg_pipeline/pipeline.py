from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Literal

from .benchmarks import BenchmarkRecord, BenchmarkRepository
from .config import ExperimentConfig, TaskConfig
from .models.base import ModelRunner, PredictionLabel
from .models._shared import normalise_label
from .preprocessing import (
    DoclingImageMode,
    extract_page_as_image,
    extract_page_as_text,
    extract_page_with_docling,
)
from .prompting import build_prompt, format_benchmark_value

LOGGER = logging.getLogger(__name__)


@dataclass
class TaskRunResult:
    task_id: str
    indicator: str
    page: int
    benchmark_year: int
    benchmark_value: float
    benchmark_unit: str
    expected_label: Optional[str]
    predicted_label: PredictionLabel
    predicted_value: Optional[str]
    match: Optional[bool]
    prompt_path: Path
    response_path: Path
    image_path: Optional[Path]
    text_path: Optional[Path]
    raw_response_path: Path
    image_paths: List[Path] = field(default_factory=list)
    combined_image_path: Optional[Path] = None

    def to_table_row(self) -> dict:
        return {
            "Task": self.task_id,
            "Indicator": self.indicator,
            "Page": self.page,
            "Benchmark": format_benchmark_value(self.benchmark_value, self.benchmark_unit),
            "Benchmark Year": self.benchmark_year,
            "Model Output": self.predicted_label,
            "Extracted Value": self.predicted_value or "",
            "Expected": self.expected_label or "",
            "Match": "yes" if self.match else ("" if self.match is None else "no"),
        }


_VALUE_KEYS = (
    "extracted_value",
    "extractedValue",
    "extracted-value",
    "expected_value",
    "expectedValue",
    "expected-value",
    "company_value",
    "companyValue",
    "company-value",
)


def _normalise_value(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    cleaned = raw.strip()
    return cleaned or None


def _extract_extracted_value(raw_response: str) -> Optional[str]:
    if not raw_response:
        return None

    def from_payload(payload: object) -> Optional[str]:
        if isinstance(payload, dict):
            for key in _VALUE_KEYS:
                if key in payload:
                    value = payload[key]
                    if isinstance(value, str):
                        return _normalise_value(value)
                    if value is not None:
                        return _normalise_value(str(value))
        return None

    stripped = raw_response.strip()
    if not stripped:
        return None

    try:
        value = from_payload(json.loads(stripped))
        if value:
            return value
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence_match:
        snippet = fence_match.group(1)
        try:
            value = from_payload(json.loads(snippet))
            if value:
                return value
        except json.JSONDecodeError:
            pass

    for brace_match in re.finditer(r"\{.*?\}", stripped, re.DOTALL):
        snippet = brace_match.group(0)
        try:
            value = from_payload(json.loads(snippet))
            if value:
                return value
        except json.JSONDecodeError:
            continue

    regex_match = re.search(
        r"(?:extracted|expected|company)[-_ ]?value\s*[:=]\s*(?:\"(?P<quoted>[^\"]+)\"|(?P<bare>[^\r\n]+))",
        stripped,
        re.IGNORECASE,
    )
    if regex_match:
        value = regex_match.group("quoted") or regex_match.group("bare")
        return _normalise_value(value)

    return None


class ExperimentRunner:
    def __init__(
        self,
        benchmark_repo: BenchmarkRepository,
        model: ModelRunner,
        artifacts_dir: Path = Path("artifacts"),
        capture_images: bool = True,
        capture_text: bool = True,
        pdf_extractor: Literal["pymupdf", "docling"] = "pymupdf",
        docling_image_mode: DoclingImageMode = "embedded",
    ) -> None:
        self.benchmark_repo = benchmark_repo
        self.model = model
        self.artifacts_dir = artifacts_dir
        self.capture_images = capture_images
        self.capture_text = capture_text
        self.pdf_extractor = pdf_extractor
        self.docling_image_mode = docling_image_mode

    def run(
        self,
        config: ExperimentConfig,
        experiment_id: Optional[str] = None,
        resume: bool = False,
    ) -> List[TaskRunResult]:
        experiment_id = experiment_id or self._derive_experiment_id(config)
        experiment_dir = self.artifacts_dir / experiment_id
        prompts_dir = experiment_dir / "prompts"
        responses_dir = experiment_dir / "responses"
        images_dir = experiment_dir / "images"
        text_dir = experiment_dir / "texts"

        for directory in [prompts_dir, responses_dir, images_dir, text_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        existing_results: Dict[str, TaskRunResult] = {}
        if resume:
            existing_results = self._load_existing_results(
                config,
                prompts_dir,
                responses_dir,
                images_dir,
                text_dir,
            )
            if existing_results:
                LOGGER.info(
                    "Loaded %s completed tasks from checkpoint in %s",
                    len(existing_results),
                    experiment_dir,
                )

        results: List[TaskRunResult] = []
        for task in config.tasks:
            if resume and task.id in existing_results:
                LOGGER.info("Skipping task %s (resume enabled)", task.id)
                results.append(existing_results[task.id])
                continue

            benchmark = self._resolve_benchmark(task)
            prompt = build_prompt(config.dataset, task, benchmark)

            prompt_path = prompts_dir / f"{task.id}.txt"
            prompt_path.write_text(prompt, encoding="utf-8")

            image_path: Optional[Path] = None
            should_capture_primary_image = self.capture_images and self.pdf_extractor != "docling"
            if should_capture_primary_image:
                try:
                    image_path = extract_page_as_image(
                        config.dataset.document,
                        task.page,
                        images_dir,
                        filename_prefix=f"{task.id}"
                    )
                except Exception as exc:  # pragma: no cover - depends on system
                    LOGGER.warning("Failed to extract image for task %s: %s", task.id, exc)

            text_path: Optional[Path] = None
            page_text: Optional[str] = None
            asset_images: List[Path] = []
            combined_image: Optional[Path] = None

            if self.pdf_extractor == "docling":
                assets_dir: Optional[Path] = None
                if self.docling_image_mode == "referenced":
                    assets_dir = images_dir / f"{task.id}_assets"
                    assets_dir.mkdir(parents=True, exist_ok=True)
                    # Clear previous remnants when rerunning tasks
                    for leftover in assets_dir.iterdir():
                        if leftover.is_file():
                            leftover.unlink()
                try:
                    artifacts = extract_page_with_docling(
                        config.dataset.document,
                        task.page,
                        markdown_dir=text_dir,
                        images_dir=assets_dir,
                        image_mode=self.docling_image_mode,
                        filename_prefix=f"{task.id}"
                    )
                    page_text = artifacts.markdown_text
                    asset_images = artifacts.image_paths
                    combined_image = artifacts.combined_image_path
                    if self.capture_text:
                        text_path = artifacts.markdown_path
                    else:
                        try:
                            artifacts.markdown_path.unlink(missing_ok=True)
                        except OSError:
                            LOGGER.debug(
                                "Failed to remove temporary markdown for task %s", task.id,
                                exc_info=True,
                            )
                except Exception as exc:  # pragma: no cover - depends on optional dependency
                    LOGGER.warning(
                        "Docling extraction failed for task %s: %s", task.id, exc
                    )
                    raise exc
                    # if self.capture_text:
                    #     try:
                    #         page_text = extract_page_as_text(
                    #             config.dataset.document, task.page
                    #         )
                    #         text_path = text_dir / f"{task.id}.txt"
                    #         text_path.write_text(page_text, encoding="utf-8")
                    #     except Exception as fallback_exc:  # pragma: no cover - system dependent
                    #         LOGGER.warning(
                    #             "Fallback text extraction failed for task %s: %s",
                    #             task.id,
                    #             fallback_exc,
                    #         )
            else:
                if self.capture_text:
                    try:
                        page_text = extract_page_as_text(config.dataset.document, task.page)
                        text_path = text_dir / f"{task.id}.txt"
                        text_path.write_text(page_text, encoding="utf-8")
                    except Exception as exc:  # pragma: no cover - depends on system
                        LOGGER.warning(
                            "Failed to extract text for task %s: %s", task.id, exc
                        )

            images_for_model: List[Path] = []
            if combined_image and combined_image.exists():
                images_for_model.append(combined_image)
            elif asset_images:
                images_for_model.extend(path for path in asset_images if path.exists())

            response = self.model.predict(
                prompt,
                page_image=image_path,
                page_text=page_text,
                page_images=images_for_model or None,
            )
            predicted_value = _extract_extracted_value(response.raw_response)
            response_path = responses_dir / f"{task.id}.json"
            response_payload = {
                "label": response.label,
                "raw_response": response.raw_response,
                "metadata": response.metadata,
                "model": getattr(self.model, "name", self.model.__class__.__name__),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "extracted_value": predicted_value,
            }
            response_path.write_text(json.dumps(response_payload, indent=2), encoding="utf-8")

            raw_text_path = responses_dir / f"{task.id}.txt"
            raw_text_path.write_text(response.raw_response, encoding="utf-8")

            expected = task.expected
            predicted = response.label
            match: Optional[bool] = None
            if expected:
                match = expected.strip().lower() == predicted.strip().lower()

            result = TaskRunResult(
                task_id=task.id,
                indicator=task.indicator,
                page=task.page,
                benchmark_year=benchmark.year,
                benchmark_value=benchmark.value,
                benchmark_unit=benchmark.unit,
                expected_label=expected,
                predicted_label=predicted,
                predicted_value=predicted_value,
                match=match,
                prompt_path=prompt_path,
                response_path=response_path,
                image_path=image_path,
                text_path=text_path,
                raw_response_path=raw_text_path,
                image_paths=asset_images,
                combined_image_path=combined_image,
            )
            results.append(result)
        self._persist_table(results, experiment_dir)
        return results

    def _persist_table(self, results: List[TaskRunResult], experiment_dir: Path) -> None:
        table_path = experiment_dir / "summary.json"
        rows = [result.to_table_row() for result in results]
        table_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def _resolve_benchmark(self, task: TaskConfig) -> BenchmarkRecord:
        if task.benchmark.value is not None:
            return BenchmarkRecord(
                indicator=task.indicator,
                unit=task.benchmark.unit,
                year=task.benchmark.year,
                value=float(task.benchmark.value),
            )
        if task.benchmark.source.lower() == "industry":
            return self.benchmark_repo.require(task.indicator, task.benchmark.year)
        raise ValueError(
            f"Unable to resolve benchmark for task {task.id}: value missing and source '{task.benchmark.source}' not supported."
        )

    def _derive_experiment_id(self, config: ExperimentConfig) -> str:
        parts = [config.dataset.company, str(config.dataset.year), getattr(self.model, "name", "model")]
        safe = "_".join(part.lower().replace(" ", "_") for part in parts)
        safe = "".join(ch for ch in safe if ch.isalnum() or ch in {"_", "-"})
        return safe

    def _load_existing_results(
        self,
        config: ExperimentConfig,
        prompts_dir: Path,
        responses_dir: Path,
        images_dir: Path,
        text_dir: Path,
    ) -> Dict[str, TaskRunResult]:
        existing: Dict[str, TaskRunResult] = {}
        for task in config.tasks:
            response_path = responses_dir / f"{task.id}.json"
            if not response_path.exists():
                continue

            try:
                payload = json.loads(response_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                LOGGER.warning(
                    "Failed to read checkpoint for task %s: %s. Task will be re-run.",
                    task.id,
                    exc,
                )
                continue

            benchmark = self._resolve_benchmark(task)

            prompt_path = prompts_dir / f"{task.id}.txt"
            if not prompt_path.exists():
                try:
                    prompt = build_prompt(config.dataset, task, benchmark)
                    prompt_path.write_text(prompt, encoding="utf-8")
                except Exception as exc:  # pragma: no cover - filesystem dependent
                    LOGGER.warning(
                        "Failed to recreate prompt for task %s during resume: %s",
                        task.id,
                        exc,
                    )

            raw_response_path = responses_dir / f"{task.id}.txt"
            raw_response = payload.get("raw_response")
            if not raw_response_path.exists() and isinstance(raw_response, str):
                try:
                    raw_response_path.write_text(raw_response, encoding="utf-8")
                except Exception as exc:  # pragma: no cover - filesystem dependent
                    LOGGER.warning(
                        "Failed to restore raw response for task %s: %s",
                        task.id,
                        exc,
                    )
            if not raw_response_path.exists():
                raw_response_path = response_path

            image_path = images_dir / f"{task.id}_p{task.page}.png"
            if not image_path.exists():
                image_path = None

            image_assets_dir = images_dir / f"{task.id}_assets"
            image_paths: List[Path] = []
            combined_image_path: Optional[Path] = None
            if image_assets_dir.exists():
                for asset_path in sorted(path for path in image_assets_dir.iterdir() if path.is_file()):
                    if asset_path.name.endswith("_combined.png") and combined_image_path is None:
                        combined_image_path = asset_path
                    else:
                        image_paths.append(asset_path)

            text_candidates = [
                text_dir / f"{task.id}.md",
                text_dir / f"{task.id}.txt",
            ]
            text_path = next((candidate for candidate in text_candidates if candidate.exists()), None)

            label_value = payload.get("label")
            if isinstance(label_value, str):
                predicted_label = normalise_label(label_value)
            else:
                predicted_label = "not found"
            raw_response_text = payload.get("raw_response")
            if isinstance(raw_response_text, str):
                extracted_value = _extract_extracted_value(raw_response_text)
            else:
                extracted_value = None
            predicted_value = None
            for key in _VALUE_KEYS:
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    predicted_value = value.strip()
                    break
            if not predicted_value:
                predicted_value = extracted_value
            expected = task.expected
            match: Optional[bool] = None
            if expected:
                match = expected.strip().lower() == predicted_label.strip().lower()

            existing[task.id] = TaskRunResult(
                task_id=task.id,
                indicator=task.indicator,
                page=task.page,
                benchmark_year=benchmark.year,
                benchmark_value=benchmark.value,
                benchmark_unit=benchmark.unit,
                expected_label=expected,
                predicted_label=predicted_label,
                predicted_value=predicted_value,
                match=match,
                prompt_path=prompt_path,
                response_path=response_path,
                image_path=image_path,
                text_path=text_path,
                raw_response_path=raw_response_path,
                image_paths=image_paths,
                combined_image_path=combined_image_path,
            )

        return existing


__all__ = ["ExperimentRunner", "TaskRunResult"]
