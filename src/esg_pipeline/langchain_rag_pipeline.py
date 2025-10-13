from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
    ) -> None:
        self.benchmark_repo = benchmark_repo
        self.model = model
        self.artifacts_dir = artifacts_dir
        self.default_config = default_config or LangchainRAGConfig()
        self._document_loader = document_loader or DocumentLoader()

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
        documents = self._document_loader.load(dataset.document, config.tasks, rag_config)
        if not documents:
            raise RuntimeError(f"No pages extracted from {dataset.document}")

        split_docs = split_documents(documents, rag_config)
        LOGGER.info("Created %s text chunks for retrieval", len(split_docs))

        vector_store = build_vector_store(split_docs, rag_config)

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

        results: List[LangchainRAGTaskResult] = []
        for task in config.tasks:
            benchmark = self._resolve_benchmark(task)
            base_prompt = build_prompt(dataset, task, benchmark)
            query = build_query(dataset.company, task)
            retrieved = retrieve_chunks(vector_store, query, rag_config)
            context_text = format_context(retrieved, rag_config)

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
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "extracted_value": predicted_value,
                "query": query,
                "chunks": [chunk.to_dict() for chunk in retrieved],
            }
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
                image_paths=[],
                combined_image_path=None,
                query=query,
                retrieved_chunks=retrieved,
            )
            results.append(result)

        artifacts.persist_summary(results)
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


__all__ = [
    "LangchainRAGConfig",
    "LangchainRAGPipeline",
    "LangchainRAGTaskResult",
]
