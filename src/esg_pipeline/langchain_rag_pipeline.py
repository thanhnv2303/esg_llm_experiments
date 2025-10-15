from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document

from .benchmarks import BenchmarkRecord, BenchmarkRepository
from .config import ExperimentConfig, TaskConfig
from .langchain_rag import (
    ArtifactWriter,
    DocumentLoader,
    LangchainRAGConfig,
    LangchainRAGTaskResult,
    build_query,
    build_vector_store,
    compose_prompt,
    derive_experiment_id,
    extract_extracted_value,
    format_context,
    RetrievedChunk,
    retrieve_chunks,
    split_documents,
)
from .models.base import ModelRunner
from .prompting import build_prompt

LOGGER = logging.getLogger(__name__)


class LangchainRAGPipeline:
    """Retrieval-Augmented Generation pipeline built on LangChain components."""

    def __init__(
        self,
        benchmark_repo: BenchmarkRepository,
        model: ModelRunner,
        artifacts_dir: Path = Path("artifacts"),
        default_config: Optional[LangchainRAGConfig] = None,
        document_loader: Optional[DocumentLoader] = None,
        chart_model: Optional[ModelRunner] = None,
    ) -> None:
        self.benchmark_repo = benchmark_repo
        self.model = model
        self.artifacts_dir = artifacts_dir
        self.default_config = default_config or LangchainRAGConfig()
        self._document_loader = document_loader or DocumentLoader()
        self.chart_model = chart_model

    def run(
        self,
        config: ExperimentConfig,
        rag_config: Optional[LangchainRAGConfig] = None,
        experiment_suffix: Optional[str] = None,
    ) -> List[LangchainRAGTaskResult]:
        rag_config = rag_config or self.default_config
        dataset = config.dataset

        LOGGER.info(
            "Preparing LangChain RAG corpus for %s (%s)",
            dataset.company,
            dataset.document,
        )
        chart_model = (self.chart_model or self.model) if rag_config.caption_charts else None
        prepared = self._document_loader.prepare(
            dataset.document,
            config.tasks,
            rag_config,
            chart_model=chart_model,
        )
        documents = prepared.documents
        if not documents:
            raise RuntimeError(f"No pages extracted from {dataset.document}")

        model_name = getattr(self.model, "name", self.model.__class__.__name__)
        experiment_id = derive_experiment_id(
            dataset.company,
            dataset.year,
            model_name,
            rag_config,
            experiment_suffix,
        )
        experiment_dir = self.artifacts_dir / experiment_id
        artifacts = ArtifactWriter(experiment_dir)

        chart_model_name: Optional[str] = None
        if chart_model is not None:
            chart_model_name = getattr(chart_model, "name", chart_model.__class__.__name__)

        chart_inventory: Dict[str, Dict[str, object]] = {
            chart_id: dict(record)
            for chart_id, record in prepared.chart_inventory.items()
        }

        page_debug_paths = self._persist_debug_artifacts(
            artifacts,
            prepared,
            rag_config,
            config,
        )

        split_docs = split_documents(documents, rag_config)
        table_records = self._collect_table_records(split_docs)
        chunk_dump_path = self._maybe_dump_chunks(split_docs, rag_config, experiment_id)
        if chunk_dump_path:
            LOGGER.info(
                "Persisted %s chunks to %s",
                len(split_docs),
                chunk_dump_path,
            )
        LOGGER.info("Created %s text chunks for retrieval", len(split_docs))

        vector_store = build_vector_store(split_docs, rag_config)

        results: List[LangchainRAGTaskResult] = []
        chunk_debug_records: Dict[str, List[RetrievedChunk]] = {}
        for task in config.tasks:
            benchmark = self._resolve_benchmark(task)
            base_prompt = build_prompt(dataset, task, benchmark)
            query = build_query(dataset.company, task)
            retrieved = retrieve_chunks(vector_store, query, rag_config)
            used_chunks = list(retrieved)
            if rag_config.top_k and len(used_chunks) > rag_config.top_k:
                used_chunks = used_chunks[: rag_config.top_k]
            chunk_debug_records[task.id] = used_chunks
            context_text = format_context(used_chunks, rag_config)

            chart_conversions = prepared.charts_by_page.get(task.page, [])
            chart_summary_path = page_debug_paths.get(task.page)
            stored_chart_images: List[Path] = []
            chart_insight_records: List[Dict[str, object]] = []
            skipped_chart_records: List[Dict[str, object]] = []

            for idx, conversion in enumerate(chart_conversions, start=1):
                if conversion.skipped:
                    skipped_chart_records.append(
                        {
                            "chart_id": conversion.chart_id,
                            "reason": conversion.skip_reason,
                            "doc_caption": conversion.doc_caption,
                            "notes": conversion.notes,
                            "ocr_text": conversion.ocr_text,
                            "page": task.page,
                        }
                    )
                    continue

                stored_path: Optional[Path] = None
                if conversion.image_path and conversion.image_path.exists():
                    stored_path = artifacts.store_chart_asset(
                        task.id,
                        conversion.image_path,
                        idx,
                        chart_id=conversion.chart_id,
                    )
                    stored_chart_images.append(stored_path)

                chart_insight_records.append(
                    {
                        "chart_id": conversion.chart_id,
                        "caption": conversion.caption or conversion.doc_caption,
                        "doc_caption": conversion.doc_caption,
                        "table_markdown": conversion.table_markdown,
                        "table_id": conversion.table_id,
                        "notes": conversion.notes,
                        "ocr_text": conversion.ocr_text,
                        "image_path": str(stored_path or conversion.image_path)
                        if (stored_path or conversion.image_path)
                        else None,
                        "page": task.page,
                    }
                )

                record = chart_inventory.get(conversion.chart_id)
                if record is None:
                    continue
                tasks_list = record.setdefault("tasks", [])
                if task.id not in tasks_list:
                    tasks_list.append(task.id)
                if stored_path:
                    stored_paths = record.setdefault("stored_image_paths", [])
                    stored_str = str(stored_path)
                    if stored_str not in stored_paths:
                        stored_paths.append(stored_str)
                if chart_summary_path:
                    record.setdefault("page_markdown_path", str(chart_summary_path))
                if chart_model_name:
                    record.setdefault("chart_model", chart_model_name)

            context_path = artifacts.write_context(task.id, context_text)
            prompt = compose_prompt(base_prompt, context_text)
            prompt_path = artifacts.write_prompt(task.id, prompt)

            LOGGER.debug(
                "Running model for task %s with %s retrieved chunks",
                task.id,
                len(retrieved),
            )
            response = self.model.predict(prompt, page_text=context_text or None)

            predicted_value = extract_extracted_value(response.raw_response)
            response_payload: Dict[str, object] = {
                "label": response.label,
                "raw_response": response.raw_response,
                "metadata": response.metadata,
                "model": model_name,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "extracted_value": predicted_value,
                "query": query,
                "chunks": [chunk.to_dict() for chunk in used_chunks],
            }
            if chart_insight_records:
                response_payload["chart_insights"] = chart_insight_records
            if skipped_chart_records:
                response_payload["skipped_charts"] = skipped_chart_records
            if stored_chart_images:
                response_payload["chart_images"] = [str(path) for path in stored_chart_images]
            if chart_summary_path:
                response_payload["chart_summary_file"] = str(chart_summary_path)
            if chart_model_name and chart_conversions:
                response_payload["chart_model"] = chart_model_name
            response_path = artifacts.write_response_payload(task.id, response_payload)
            raw_text_path = artifacts.write_raw_response(task.id, response.raw_response)

            expected = task.expected
            predicted = response.label
            match: Optional[bool] = None
            if expected:
                match = expected.strip().lower() == predicted.strip().lower()

            result = LangchainRAGTaskResult(
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
                image_path=None,
                text_path=context_path,
                raw_response_path=raw_text_path,
                image_paths=stored_chart_images,
                combined_image_path=None,
                query=query,
                retrieved_chunks=used_chunks,
                chart_insights=chart_insight_records,
                chart_summary_path=chart_summary_path,
            )
            results.append(result)

        artifacts.persist_summary(results)
        chart_records = self._serialise_chart_inventory(chart_inventory)
        artifacts.write_chart_table_summary(
            chart_records=chart_records,
            table_records=table_records,
        )
        self._persist_chunk_debug(artifacts, chunk_debug_records, rag_config)
        return results

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

    def _persist_debug_artifacts(
        self,
        artifacts: ArtifactWriter,
        prepared_documents,
        config: LangchainRAGConfig,
        experiment_config: ExperimentConfig,
    ) -> Dict[int, Path]:
        debug_dir = artifacts.experiment_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        run_config_path = debug_dir / "run_config.json"
        run_config_payload = {
            "rag_config": dataclasses.asdict(config),
            "experiment": dataclasses.asdict(experiment_config),
        }
        run_config_path.write_text(
            json.dumps(run_config_payload, indent=2, default=str),
            encoding="utf-8",
        )

        page_paths: Dict[int, Path] = {}

        if config.debug_store_pages and prepared_documents.page_markdowns:
            pages_dir = debug_dir / "pages"
            pages_dir.mkdir(parents=True, exist_ok=True)
            for page, markdown in prepared_documents.page_markdowns.items():
                path = pages_dir / f"page_{page:04d}.md"
                path.write_text(markdown, encoding="utf-8")
                page_paths[page] = path

        if config.debug_store_tables and prepared_documents.table_documents:
            tables_dir = debug_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            for doc in prepared_documents.table_documents:
                metadata = dict(getattr(doc, "metadata", {}) or {})
                table_id = metadata.get("table_id") or "table"
                chunk_index = metadata.get("table_chunk_index")
                try:
                    suffix = f"_chunk{int(chunk_index):02d}" if chunk_index is not None else ""
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    suffix = ""
                filename = f"{table_id}{suffix}.md"
                path = tables_dir / filename
                path.write_text(doc.page_content, encoding="utf-8")

        if (config.debug_store_chart_tables or config.debug_store_captions) and prepared_documents.charts_by_page:
            charts_dir = debug_dir / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)
            for conversions in prepared_documents.charts_by_page.values():
                for conversion in conversions:
                    base = conversion.chart_id
                    if config.debug_store_chart_tables and conversion.table_markdown:
                        table_path = charts_dir / f"{base}_table.md"
                        table_path.write_text(conversion.table_markdown, encoding="utf-8")
                    if config.debug_store_captions and conversion.raw_response:
                        raw_path = charts_dir / f"{base}_raw.txt"
                        raw_path.write_text(conversion.raw_response, encoding="utf-8")

        if config.debug_store_images and getattr(prepared_documents, "snapshots", None):
            images_dir = debug_dir / "images"
            copied: set[Path] = set()
            for page, snapshot in prepared_documents.snapshots.items():
                if not snapshot.image_paths:
                    continue
                page_dir = images_dir / f"page_{page:04d}"
                page_dir.mkdir(parents=True, exist_ok=True)
                for source in snapshot.image_paths:
                    if not source.exists():
                        continue
                    destination = page_dir / source.name
                    if destination in copied:
                        continue
                    shutil.copy2(source, destination)
                    copied.add(destination)

            if prepared_documents.charts_by_page:
                charts_img_dir = images_dir / "charts"
                charts_img_dir.mkdir(parents=True, exist_ok=True)
                for conversions in prepared_documents.charts_by_page.values():
                    for conversion in conversions:
                        source = conversion.image_path
                        if not source or not source.exists():
                            continue
                        destination = charts_img_dir / source.name
                        if destination in copied:
                            continue
                        shutil.copy2(source, destination)
                        copied.add(destination)

        return page_paths

    def _persist_chunk_debug(
        self,
        artifacts: ArtifactWriter,
        chunk_map: Dict[str, List[RetrievedChunk]],
        config: LangchainRAGConfig,
    ) -> None:
        if not chunk_map or not config.debug_store_chunks:
            return

        debug_dir = artifacts.experiment_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir = debug_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        for task_id, chunks in chunk_map.items():
            lines: List[str] = [f"# Task {task_id}"]
            for index, chunk in enumerate(chunks, start=1):
                header_bits = [f"Chunk {index}"]
                if chunk.page is not None:
                    header_bits.append(f"page {chunk.page}")
                if chunk.score is not None:
                    header_bits.append(f"score {chunk.score:.4f}")
                header = " | ".join(header_bits)
                metadata_json = json.dumps(chunk.metadata, indent=2, default=str) if chunk.metadata else "{}"
                lines.append(f"## {header}")
                lines.append("Metadata:")
                lines.append("```json")
                lines.append(metadata_json)
                lines.append("```")
                lines.append("")
                lines.append(chunk.content)
                lines.append("")
            chunk_path = chunks_dir / f"{task_id}.md"
            chunk_path.write_text("\n".join(lines), encoding="utf-8")

    def _maybe_dump_chunks(
        self,
        chunks: List[Document],
        config: LangchainRAGConfig,
        experiment_id: str,
    ) -> Optional[Path]:
        if not chunks:
            return None

        target_dir: Optional[Path]
        if config.chunk_dump_dir:
            base_dir = Path(config.chunk_dump_dir).expanduser()
            target_dir = base_dir / experiment_id
            target_dir.mkdir(parents=True, exist_ok=True)
        elif config.save_chunks_to_temp:
            temp_path = tempfile.mkdtemp(prefix=f"{experiment_id}_chunks_")
            target_dir = Path(temp_path)
        else:
            return None

        for idx, doc in enumerate(chunks):
            payload = {
                "index": idx,
                "page_content": doc.page_content,
                "metadata": dict(getattr(doc, "metadata", {})),
            }
            chunk_path = target_dir / f"chunk_{idx:04d}.json"
            chunk_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return target_dir

    def _collect_table_records(
        self,
        documents: Sequence[Document],
    ) -> List[Dict[str, object]]:
        if not documents:
            return []

        tables: Dict[str, Dict[str, object]] = {}
        for doc in documents:
            metadata = dict(getattr(doc, "metadata", {}) or {})
            if metadata.get("type") != "table":
                continue
            table_id = metadata.get("table_id")
            if not table_id:
                continue

            page_meta = metadata.get("page")
            page: Optional[int] = None
            if isinstance(page_meta, int):
                page = page_meta + 1
            elif isinstance(page_meta, str) and page_meta.isdigit():
                page = int(page_meta) + 1

            chunk_index = metadata.get("table_chunk_index")
            chunk_count = metadata.get("table_chunk_count")
            try:
                chunk_index = int(chunk_index) if chunk_index is not None else None
            except (TypeError, ValueError):  # pragma: no cover - defensive
                chunk_index = None
            try:
                chunk_count = int(chunk_count) if chunk_count is not None else None
            except (TypeError, ValueError):  # pragma: no cover - defensive
                chunk_count = None

            record = tables.get(table_id)
            if record is None:
                record = {
                    "table_id": table_id,
                    "page": page,
                    "title": metadata.get("table_title") or metadata.get("table_caption"),
                    "caption": metadata.get("table_caption"),
                    "context": metadata.get("table_context"),
                    "chunk_count": chunk_count,
                    "chunks": [],
                    "header": metadata.get("table_header"),
                    "source": metadata.get("source"),
                }
                tables[table_id] = record
            else:
                if record.get("page") is None and page is not None:
                    record["page"] = page
                if record.get("chunk_count") is None and chunk_count is not None:
                    record["chunk_count"] = chunk_count

            record_chunk = {
                "chunk_index": chunk_index,
                "markdown": doc.page_content,
            }
            record["chunks"].append(record_chunk)

        table_records = list(tables.values())
        for record in table_records:
            chunks = record.get("chunks", [])
            if isinstance(chunks, list) and chunks:
                chunks.sort(key=lambda item: (item.get("chunk_index") or 0))
        table_records.sort(key=lambda item: (item.get("page") or 0, item.get("table_id") or ""))
        return table_records

    @staticmethod
    def _serialise_chart_inventory(
        chart_inventory: Dict[str, Dict[str, object]],
    ) -> List[Dict[str, object]]:
        if not chart_inventory:
            return []

        records = []
        for record in chart_inventory.values():
            serialised = dict(record)
            for key in ("stored_image_paths", "tasks"):
                value = serialised.get(key)
                if isinstance(value, list) and value:
                    serialised[key] = list(dict.fromkeys(value))
            records.append(serialised)

        records.sort(key=lambda item: (item.get("page") or 0, item.get("chart_id") or ""))
        return records


__all__ = [
    "LangchainRAGConfig",
    "LangchainRAGPipeline",
    "LangchainRAGTaskResult",
]
