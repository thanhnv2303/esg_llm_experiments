from __future__ import annotations

import hashlib
import json
import logging
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from .benchmarks import BenchmarkRecord, BenchmarkRepository
from .config import ExperimentConfig, TaskConfig
from .models.base import ModelRunner
from .pipeline import TaskRunResult
from .prompting import build_prompt

LOGGER = logging.getLogger(__name__)


class HashingEmbeddings(Embeddings):
    """Light-weight embedding function based on token hashing.

    This avoids heavyweight model downloads while still providing
    deterministic similarity scoring for retrieval experiments.
    """

    def __init__(self, dimension: int = 768) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dimension
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in _tokenise(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimension
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0.0:
            vector /= norm
        return vector.astype(np.float32).tolist()


def _tokenise(text: str) -> Iterable[str]:
    current: List[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            yield "".join(current)
            current.clear()
    if current:
        yield "".join(current)


@dataclass
class RetrievedChunk:
    content: str
    page: Optional[int]
    score: Optional[float]
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "content": self.content,
            "page": self.page,
        }
        if self.score is not None:
            payload["score"] = _serialise_json(self.score)
        if self.metadata:
            payload["metadata"] = {
                key: _serialise_json(value) for key, value in self.metadata.items()
            }
        return payload


@dataclass
class LangchainRAGConfig:
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
        return (
            f"chunk{self.chunk_size}-overlap{self.chunk_overlap}-k{self.top_k}" +
            ("-mmr" if self.use_mmr else "-sim") +
            f"-dim{self.embedding_dimension}"
        )


@dataclass
class LangchainRAGTaskResult(TaskRunResult):
    query: str = ""
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)


class LangchainRAGPipeline:
    """RAG experiment runner that pairs ESG tasks with LangChain retrieval."""

    def __init__(
        self,
        benchmark_repo: BenchmarkRepository,
        model: ModelRunner,
        artifacts_dir: Path = Path("artifacts"),
        default_config: Optional[LangchainRAGConfig] = None,
    ) -> None:
        self.benchmark_repo = benchmark_repo
        self.model = model
        self.artifacts_dir = artifacts_dir
        self.default_config = default_config or LangchainRAGConfig()
        self._raw_documents: Optional[List[Document]] = None
        self._raw_document_source: Optional[Path] = None

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
        documents = self._load_documents(dataset.document, config.tasks, rag_config)
        if not documents:
            raise RuntimeError(f"No pages extracted from {dataset.document}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=rag_config.chunk_size,
            chunk_overlap=rag_config.chunk_overlap,
            add_start_index=True,
        )
        split_docs = splitter.split_documents(documents)
        LOGGER.info("Created %s text chunks for retrieval", len(split_docs))

        embeddings = HashingEmbeddings(dimension=rag_config.embedding_dimension)
        vector_store = FAISS.from_documents(split_docs, embeddings)

        experiment_id = self._derive_experiment_id(dataset.company, dataset.year, rag_config, experiment_suffix)
        experiment_dir = self.artifacts_dir / experiment_id
        prompts_dir = experiment_dir / "prompts"
        responses_dir = experiment_dir / "responses"
        contexts_dir = experiment_dir / "contexts"

        for directory in (prompts_dir, responses_dir, contexts_dir):
            directory.mkdir(parents=True, exist_ok=True)

        results: List[LangchainRAGTaskResult] = []
        for task in config.tasks:
            benchmark = self._resolve_benchmark(task)
            base_prompt = build_prompt(dataset, task, benchmark)
            query = self._build_query(dataset.company, task)
            retrieved = self._retrieve(vector_store, query, rag_config)
            context_text = self._format_context(retrieved, rag_config)

            context_path = contexts_dir / f"{task.id}.md"
            context_path.write_text(context_text, encoding="utf-8")

            prompt = self._compose_prompt(base_prompt, context_text)
            prompt_path = prompts_dir / f"{task.id}.txt"
            prompt_path.write_text(prompt, encoding="utf-8")

            LOGGER.debug("Running model for task %s with %s retrieved chunks", task.id, len(retrieved))
            response = self.model.predict(prompt, page_text=context_text or None)

            predicted_value = _extract_extracted_value(response.raw_response)
            response_payload = {
                "label": response.label,
                "raw_response": response.raw_response,
                "metadata": response.metadata,
                "model": getattr(self.model, "name", self.model.__class__.__name__),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "extracted_value": predicted_value,
                "query": query,
                "chunks": [chunk.to_dict() for chunk in retrieved],
            }
            response_path = responses_dir / f"{task.id}.json"
            response_path.write_text(json.dumps(response_payload, indent=2), encoding="utf-8")

            raw_text_path = responses_dir / f"{task.id}.txt"
            raw_text_path.write_text(response.raw_response, encoding="utf-8")

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

        self._persist_table(results, experiment_dir)
        return results

    def _load_documents(
        self,
        document_path: Path,
        tasks: Sequence[TaskConfig],
        rag_config: LangchainRAGConfig,
    ) -> List[Document]:
        document_path = document_path.expanduser().resolve()
        if self._raw_documents is not None and self._raw_document_source == document_path:
            documents = self._raw_documents
        else:
            loader = PyPDFLoader(str(document_path))
            documents = loader.load()
            self._raw_documents = documents
            self._raw_document_source = document_path

        if not rag_config.restrict_to_task_pages:
            return list(documents)

        requested_pages = self._expand_task_pages(tasks, rag_config.page_padding)
        filtered = [doc for doc in documents if _page_one_based(doc) in requested_pages]
        return filtered

    def _expand_task_pages(self, tasks: Sequence[TaskConfig], padding: int) -> Iterable[int]:
        pages: set[int] = set()
        for task in tasks:
            base = max(1, task.page)
            for delta in range(-padding, padding + 1):
                pages.add(max(1, base + delta))
        return sorted(pages)

    def _retrieve(
        self,
        vector_store: FAISS,
        query: str,
        rag_config: LangchainRAGConfig,
    ) -> List[RetrievedChunk]:
        if not query.strip():
            return []

        if rag_config.use_mmr:
            base_docs = vector_store.max_marginal_relevance_search(
                query,
                k=rag_config.top_k,
                fetch_k=max(rag_config.mmr_fetch_k, rag_config.top_k),
                lambda_mult=rag_config.mmr_lambda,
            )
            pairs = [(doc, None) for doc in base_docs]
        else:
            base_docs = vector_store.similarity_search_with_score(query, k=rag_config.top_k)
            pairs = base_docs

        chunks: List[RetrievedChunk] = []
        for item in pairs:
            if isinstance(item, tuple):
                doc, score = item
            else:  # pragma: no cover - compatibility
                doc, score = item, None
            chunks.append(self._chunk_from_document(doc, score))
        return chunks

    def _chunk_from_document(self, doc: Document, score: Optional[float]) -> RetrievedChunk:
        metadata = dict(doc.metadata)
        page = _page_one_based(doc)
        content = doc.page_content.strip()
        preview = textwrap.shorten(content, width=600, placeholder=" ...")
        return RetrievedChunk(
            content=preview,
            page=page,
            score=score,
            metadata=metadata,
        )

    def _format_context(
        self,
        chunks: List[RetrievedChunk],
        rag_config: LangchainRAGConfig,
    ) -> str:
        if not chunks:
            return "No relevant context retrieved."

        parts: List[str] = []
        for index, chunk in enumerate(chunks, start=1):
            header_bits = [f"Chunk {index}"]
            if chunk.page is not None:
                header_bits.append(f"page {chunk.page}")
            if chunk.score is not None:
                header_bits.append(f"score {chunk.score:.4f}")
            header = " | ".join(header_bits)
            parts.append(f"## {header}\n{chunk.content.strip()}")

        context = rag_config.context_separator.join(parts)
        if rag_config.max_context_characters and len(context) > rag_config.max_context_characters:
            context = context[: rag_config.max_context_characters] + "\n\n[truncated]"
        return context

    def _compose_prompt(self, base_prompt: str, context: str) -> str:
        return f"{base_prompt}\n\nContext provided by retrieval:\n{context}\n\nAnswer strictly using only this context."

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

    def _build_query(self, company: str, task: TaskConfig) -> str:
        terms = [company, task.indicator]
        terms.extend(task.prompt_overrides.indicator_synonyms)
        if task.expected:
            terms.append(task.expected)
        return " ".join(term.strip() for term in terms if term and term.strip())

    def _compose_summary_rows(self, results: Iterable[LangchainRAGTaskResult]) -> List[Dict[str, object]]:
        return [result.to_table_row() for result in results]

    def _persist_table(self, results: List[LangchainRAGTaskResult], experiment_dir: Path) -> None:
        table_path = experiment_dir / "summary.json"
        rows = self._compose_summary_rows(results)
        table_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def _derive_experiment_id(
        self,
        company: str,
        year: int,
        rag_config: LangchainRAGConfig,
        suffix: Optional[str],
    ) -> str:
        base = f"{company}_{year}_{getattr(self.model, 'name', self.model.__class__.__name__)}"
        safe = "_".join(part.lower().replace(" ", "_") for part in base.split("_"))
        safe = "".join(char for char in safe if char.isalnum() or char in {"_", "-"})
        descriptor = rag_config.describe()
        if suffix:
            descriptor = f"{descriptor}-{suffix}"
        return f"{safe}_{descriptor}"


def _page_one_based(doc: Document) -> Optional[int]:
    page = doc.metadata.get("page")
    if isinstance(page, int):
        return page + 1
    if isinstance(page, str) and page.isdigit():
        return int(page) + 1
    return None


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


def _extract_extracted_value(raw_response: str) -> Optional[str]:
    if not raw_response:
        return None

    stripped = raw_response.strip()
    if not stripped:
        return None

    for key in _VALUE_KEYS:
        marker = f"{key}:"
        lower = key.replace("_", " ").replace("-", " ")
        if marker in stripped:
            fragment = stripped.split(marker, 1)[-1]
            return fragment.splitlines()[0].strip().strip(",")
        marker = f"{lower}:"
        if marker in stripped.lower():
            fragment = stripped.lower().split(marker, 1)[-1]
            return fragment.splitlines()[0].strip().strip(",")

    return None


def _serialise_json(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.integer):  # type: ignore[attr-defined]
        return int(value)
    if isinstance(value, np.floating):  # type: ignore[attr-defined]
        return float(value)
    return str(value)


__all__ = [
    "LangchainRAGConfig",
    "LangchainRAGPipeline",
    "LangchainRAGTaskResult",
]
