#!/usr/bin/env python3
"""
Utility script to purge LightRAG storages (graph, vector, KV, doc status) for the
`rag-anything` pipeline. This is useful when you need a clean slate between
experiments while reusing the same backing databases.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.esg_pipeline.rag_anything_pipeline import RAGAnythingPipelineConfig


async def _noop_llm(*_args, **_kwargs) -> str:
    """Stub LLM callable required by LightRAG initialisation."""

    return ""


def _build_embedding_stub(dimension: int, max_tokens: int | None) -> EmbeddingFunc:
    """Create a minimal EmbeddingFunc that returns zero vectors of the right shape."""

    async def _stub(texts: Iterable[str]) -> np.ndarray:
        count = sum(1 for _ in texts)
        if count == 0:
            return np.zeros((0, dimension), dtype=np.float32)
        return np.zeros((count, dimension), dtype=np.float32)

    return EmbeddingFunc(embedding_dim=dimension, func=_stub, max_token_size=max_tokens)


async def _cleanup_storages(
    config_path: Path, workspace: str | None, dry_run: bool
) -> list[Tuple[str, str, str]]:
    pipeline_config = RAGAnythingPipelineConfig.from_json(config_path)
    rag_config = pipeline_config.build_rag_config()

    working_dir = Path(rag_config.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    lightrag_kwargs = dict(pipeline_config.lightrag_options)
    if workspace:
        lightrag_kwargs["workspace"] = workspace

    vector_kwargs = lightrag_kwargs.setdefault("vector_db_storage_cls_kwargs", {})
    vector_kwargs.setdefault("cosine_better_than_threshold", 0.18)

    embedding_cfg = pipeline_config.embedding
    embedding_dim = embedding_cfg.embedding_dim if embedding_cfg else 1536
    max_tokens = embedding_cfg.max_token_size if embedding_cfg else None
    embedding = _build_embedding_stub(embedding_dim, max_tokens)

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=_noop_llm,
        embedding_func=embedding,
        **lightrag_kwargs,
    )

    await rag.initialize_storages()

    storages = [
        ("kv.full_docs", rag.full_docs),
        ("kv.text_chunks", rag.text_chunks),
        ("kv.full_entities", rag.full_entities),
        ("kv.full_relations", rag.full_relations),
        ("kv.llm_cache", rag.llm_response_cache),
        ("vdb.entities", rag.entities_vdb),
        ("vdb.relationships", rag.relationships_vdb),
        ("vdb.chunks", rag.chunks_vdb),
        ("graph.chunk_entity_relation", rag.chunk_entity_relation_graph),
        ("doc_status", rag.doc_status),
    ]

    results: list[Tuple[str, str, str]] = []
    for name, storage in storages:
        dropper = getattr(storage, "drop", None)
        if dropper is None:
            results.append((name, "skipped", "drop not supported"))
            continue

        if dry_run:
            results.append((name, "skipped", "dry run"))
            continue

        try:
            outcome = await dropper()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            results.append((name, "error", str(exc)))
            continue

        if isinstance(outcome, dict):
            status = outcome.get("status", "ok")
            message = outcome.get("message", "")
        else:
            status = "ok"
            message = str(outcome) if outcome else ""
        results.append((name, status, message))

    await rag.finalize_storages()

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup LightRAG persistent storages")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("local/configs/rag_anything_distributed.json"),
        help="Path to the rag-anything JSON configuration",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Override the LightRAG workspace (defaults to config or env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which storages would be purged without deleting data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = asyncio.run(_cleanup_storages(args.config, args.workspace, args.dry_run))

    header = ("storage", "status", "message")
    widths = [max(len(row[i]) for row in ([header] + results)) for i in range(3)]

    print(" | ".join(h.ljust(widths[idx]) for idx, h in enumerate(header)))
    print("-+-".join("-" * w for w in widths))

    for storage, status, message in results:
        row = (storage.ljust(widths[0]), status.ljust(widths[1]), message.ljust(widths[2]))
        print(" | ".join(row))


if __name__ == "__main__":
    main()
