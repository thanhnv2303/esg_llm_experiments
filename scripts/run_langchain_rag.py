#!/usr/bin/env python3
"""CLI utility to run ESG experiments with the LangChain RAG pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from esg_pipeline import (  # noqa: E402
    BenchmarkRepository,
    CerebrasModel,
    DummyModel,
    GoogleGenerativeModel,
    GroqModel,
    LangchainRAGConfig,
    LangchainRAGPipeline,
    OpenAICompatibleModel,
    RAGAnythingModel,
    RAGAnythingPipelineConfig,
    load_experiment,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ESG tasks with the LangChain RAG pipeline.")
    parser.add_argument("--experiment", required=True, type=Path, help="Path to experiment JSON file.")
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=ROOT_DIR / "benchmarks" / "industry_indicators.csv",
        help="CSV file providing benchmark values.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ROOT_DIR / "artifacts",
        help="Directory where prompts, responses, and contexts will be stored.",
    )
    parser.add_argument(
        "--experiment-suffix",
        type=str,
        default=None,
        help="Optional suffix appended to the derived experiment identifier.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    # Model options (subset mirrored from main CLI)
    parser.add_argument(
        "--model-backend",
        choices=["dummy", "openai", "groq", "google", "cerebras", "rag-anything"],
        default="dummy",
        help="Model backend that generates the final answer.",
    )
    parser.add_argument(
        "--dummy-label",
        choices=["higher", "lower", "equal", "not found"],
        default="not found",
        help="Prediction label returned by the dummy backend.",
    )
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com"),
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument(
        "--extra-header",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Additional HTTP header (may be provided multiple times).",
    )
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=60.0)

    parser.add_argument("--groq-model", type=str, default="llama-3.2-11b-vision-preview")
    parser.add_argument(
        "--groq-api-base",
        type=str,
        default=os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai"),
    )
    parser.add_argument("--groq-api-key", type=str, default=None)
    parser.add_argument("--groq-api-key-env", type=str, default="GROQ_API_KEY")

    parser.add_argument("--cerebras-model", type=str, default="llama3.1-8b")
    parser.add_argument(
        "--cerebras-api-base",
        type=str,
        default=os.environ.get("CEREBRAS_API_BASE", "https://api.cerebras.ai"),
    )
    parser.add_argument("--cerebras-api-key", type=str, default=None)
    parser.add_argument("--cerebras-api-key-env", type=str, default="CEREBRAS_API_KEY")

    parser.add_argument("--google-model", type=str, default="gemini-2.5-flash")
    parser.add_argument(
        "--google-api-base",
        type=str,
        default=os.environ.get("GOOGLE_API_BASE", "https://generativelanguage.googleapis.com"),
    )
    parser.add_argument("--google-api-key", type=str, default=None)
    parser.add_argument("--google-api-key-env", type=str, default="GOOGLE_API_KEY")
    parser.add_argument("--google-top-k", type=int, default=None)
    parser.add_argument("--google-max-output-tokens", type=int, default=None)
    parser.add_argument(
        "--google-safety",
        action="append",
        default=None,
        metavar="CATEGORY=THRESHOLD",
        help="Safety setting override for Google Generative AI; can be repeated.",
    )

    parser.add_argument(
        "--rag-config",
        type=Path,
        default=None,
        help="Pipeline configuration JSON when using the rag-anything backend.",
    )

    # LangChain RAG parameters
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--embedding-dimension", type=int, default=768)
    parser.add_argument(
        "--embedding-backend",
        type=str,
        default="hash",
        choices=[
            "hash",
            "hashing",
            "huggingface",
            "hf",
            "sentence-transformers",
            "openai",
            "openai-embeddings",
            "azure-openai",
        ],
        help="Embedding backend used to build the LangChain vector store.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model identifier passed to the selected backend.",
    )
    parser.add_argument(
        "--embedding-option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Additional keyword argument forwarded to the embedding backend.",
    )
    parser.add_argument(
        "--include-all-pages",
        action="store_true",
        help="Index every PDF page instead of limiting to task pages.",
    )
    parser.add_argument("--page-padding", type=int, default=0)
    parser.add_argument(
        "--use-mmr",
        action="store_true",
        help="Use maximal marginal relevance retrieval instead of pure similarity.",
    )
    parser.add_argument("--mmr-lambda", type=float, default=0.5)
    parser.add_argument("--mmr-fetch-k", type=int, default=12)
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument(
        "--context-separator",
        type=str,
        default="\n\n---\n\n",
        help="Separator string between retrieved chunks in the context file.",
    )
    parser.add_argument(
        "--save-chunks",
        action="store_true",
        help="Persist chunked documents to a temporary directory for inspection.",
    )
    parser.add_argument(
        "--chunk-dump-dir",
        type=Path,
        default=None,
        help="Optional directory where chunked documents should be saved. A subfolder per run will be created inside this path.",
    )
    parser.add_argument(
        "--disable-table-extraction",
        action="store_true",
        help="Skip table extraction and rely solely on text chunks.",
    )
    parser.add_argument(
        "--table-row-chunk-size",
        type=int,
        default=20,
        help="Maximum number of table rows per chunk when linearising large tables.",
    )
    parser.add_argument(
        "--table-disable-docling",
        action="store_true",
        help="Disable Docling even if installed when extracting tables.",
    )
    parser.add_argument(
        "--caption-charts",
        dest="caption_charts",
        action="store_true",
        default=True,
        help="Summarise chart and figure images using the model before answering each task (default: enabled).",
    )
    parser.add_argument(
        "--no-caption-charts",
        dest="caption_charts",
        action="store_false",
        help="Disable chart summarisation before retrieval/answering.",
    )
    parser.add_argument(
        "--chart-caption-prompt",
        type=str,
        default=None,
        help="Override the prompt used when asking the model to interpret charts.",
    )
    parser.add_argument(
        "--chart-to-table-prompt",
        type=str,
        default=None,
        help="Override the JSON table conversion prompt used for chart images.",
    )
    parser.add_argument(
        "--chart-caption-max-images",
        type=int,
        default=-1,
        help="Maximum number of images per page sent to the captioning model (<=0 means no limit).",
    )
    parser.add_argument(
        "--chart-caption-disable-docling",
        action="store_true",
        help="Skip Docling extraction when gathering chart images (falls back to full-page renders).",
    )
    parser.add_argument(
        "--parse-cache-dir",
        type=Path,
        default=None,
        help="Directory where parsed page artefacts should be cached across runs (set to empty to disable).",
    )
    parser.add_argument(
        "--no-parse-cache",
        action="store_true",
        help="Disable caching of parsed pages and Docling outputs.",
    )
    parser.add_argument(
        "--debug-store-pages",
        dest="debug_store_pages",
        action="store_true",
        help="Persist page-level Markdown representations for debugging.",
    )
    parser.add_argument(
        "--no-debug-store-pages",
        dest="debug_store_pages",
        action="store_false",
        help="Disable page-level Markdown debug output.",
    )
    parser.add_argument(
        "--debug-store-tables",
        dest="debug_store_tables",
        action="store_true",
        help="Persist extracted table Markdown blocks for debugging.",
    )
    parser.add_argument(
        "--no-debug-store-tables",
        dest="debug_store_tables",
        action="store_false",
        help="Disable table Markdown debug output.",
    )
    parser.add_argument(
        "--debug-store-chart-tables",
        dest="debug_store_chart_tables",
        action="store_true",
        help="Persist chart-to-table conversions for debugging.",
    )
    parser.add_argument(
        "--no-debug-store-chart-tables",
        dest="debug_store_chart_tables",
        action="store_false",
        help="Disable chart-to-table debug output.",
    )
    parser.add_argument(
        "--debug-store-captions",
        dest="debug_store_captions",
        action="store_true",
        help="Persist raw model responses from chart interpretation.",
    )
    parser.add_argument(
        "--no-debug-store-captions",
        dest="debug_store_captions",
        action="store_false",
        help="Disable storage of raw chart model responses.",
    )
    parser.add_argument(
        "--debug-store-chunks",
        dest="debug_store_chunks",
        action="store_true",
        help="Persist retrieved chunk payloads (Markdown + metadata) under debug/chunks/.",
    )
    parser.add_argument(
        "--no-debug-store-chunks",
        dest="debug_store_chunks",
        action="store_false",
        help="Disable chunk debug output.",
    )
    parser.add_argument(
        "--debug-store-images",
        dest="debug_store_images",
        action="store_true",
        help="Copy extracted page/chart images into debug/images/ for inspection.",
    )
    parser.add_argument(
        "--no-debug-store-images",
        dest="debug_store_images",
        action="store_false",
        help="Disable image debug output.",
    )
    parser.add_argument(
        "--debug-all",
        dest="debug_all",
        action="store",
        help="Enable save images,charts,tables,captions,pages, in {artifacts}/debug/ folder.",
    )

    parser.set_defaults(
        debug_store_pages=None,
        debug_store_tables=None,
        debug_store_chart_tables=None,
        debug_store_captions=None,
        debug_store_chunks=None,
        debug_store_images=None,
    )

    return parser.parse_args()


def parse_extra_headers(values: list[str] | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if not values:
        return headers
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid header '{item}'. Expected KEY=VALUE format.")
        key, value = item.split("=", 1)
        headers[key.strip()] = value.strip()
    return headers


def parse_google_safety_settings(values: list[str] | None) -> list[dict[str, str]]:
    settings: list[dict[str, str]] = []
    if not values:
        return settings
    for item in values:
        if "=" not in item:
            raise ValueError(
                f"Invalid safety setting '{item}'. Expected CATEGORY=THRESHOLD format."
            )
        category, threshold = item.split("=", 1)
        settings.append({"category": category.strip(), "threshold": threshold.strip()})
    return settings


def parse_key_value_pairs(values: list[str] | None) -> dict[str, object]:
    pairs: dict[str, object] = {}
    if not values:
        return pairs
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid option '{item}'. Expected KEY=VALUE format.")
        key, value = item.split("=", 1)
        pairs[key.strip()] = _coerce_value(value.strip())
    return pairs


def _coerce_value(value: str) -> object:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def build_model(args: argparse.Namespace, artifacts_dir: Path) -> object:
    if args.model_backend == "dummy":
        return DummyModel(default_label=args.dummy_label)

    if args.model_backend == "openai":
        api_key = args.api_key or os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI-compatible backend requires an API key.")
        headers = parse_extra_headers(args.extra_header)
        return OpenAICompatibleModel(
            model=args.openai_model,
            api_base=args.api_base,
            api_key=api_key,
            temperature=args.temperature,
            top_p=args.top_p if args.top_p is not None else 1.0,
            max_tokens=args.max_tokens,
            timeout_seconds=args.request_timeout,
            extra_headers=headers,
            max_retries=args.retry_attempts,
            retry_wait_seconds=args.retry_wait,
        )

    if args.model_backend == "groq":
        api_key = args.groq_api_key or os.environ.get(args.groq_api_key_env) or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq backend requires an API key.")
        headers = parse_extra_headers(args.extra_header)
        return GroqModel(
            model=args.groq_model,
            api_key=api_key,
            api_base=args.groq_api_base,
            temperature=args.temperature,
            top_p=args.top_p if args.top_p is not None else 1.0,
            max_tokens=args.max_tokens,
            timeout_seconds=args.request_timeout,
            extra_headers=headers,
            max_retries=args.retry_attempts,
            retry_wait_seconds=args.retry_wait,
        )

    if args.model_backend == "google":
        api_key = (
            args.google_api_key
            or os.environ.get(args.google_api_key_env)
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            raise ValueError("Google Generative AI backend requires an API key.")
        headers = parse_extra_headers(args.extra_header)
        safety_settings = parse_google_safety_settings(args.google_safety)
        max_output_tokens = args.google_max_output_tokens if args.google_max_output_tokens is not None else args.max_tokens
        return GoogleGenerativeModel(
            model=args.google_model,
            api_key=api_key,
            api_base=args.google_api_base,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.google_top_k,
            max_output_tokens=max_output_tokens,
            timeout_seconds=args.request_timeout,
            safety_settings=safety_settings or None,
            extra_headers=headers,
            max_retries=args.retry_attempts,
            retry_wait_seconds=args.retry_wait,
        )

    if args.model_backend == "cerebras":
        api_key = (
            args.cerebras_api_key
            or os.environ.get(args.cerebras_api_key_env)
            or os.environ.get("CEREBRAS_API_KEY")
        )
        if not api_key:
            raise ValueError("Cerebras backend requires an API key.")
        headers = parse_extra_headers(args.extra_header)
        return CerebrasModel(
            model=args.cerebras_model,
            api_key=api_key,
            api_base=args.cerebras_api_base,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            timeout_seconds=args.request_timeout,
            extra_headers=headers,
            max_retries=args.retry_attempts,
            retry_wait_seconds=args.retry_wait,
        )

    if args.model_backend == "rag-anything":
        if args.rag_config is None:
            raise ValueError("RAG-Anything backend requires --rag-config pointing to the configuration JSON.")
        rag_config = RAGAnythingPipelineConfig.from_json(args.rag_config)
        return RAGAnythingModel(config=rag_config, artifacts_dir=artifacts_dir)

    raise ValueError(f"Unsupported model backend: {args.model_backend}")


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    load_dotenv(ROOT_DIR / ".env", override=False)

    benchmark_repo = BenchmarkRepository(args.benchmarks)
    experiment = load_experiment(args.experiment)

    model = build_model(args, artifacts_dir=args.artifacts)

    if hasattr(model, "prepare_dataset"):
        model.prepare_dataset(experiment)

    default_rag_config = LangchainRAGConfig()

    embedding_kwargs = parse_key_value_pairs(args.embedding_option)

    chart_prompt = args.chart_caption_prompt or default_rag_config.chart_caption_prompt
    chart_to_table_prompt = args.chart_to_table_prompt or default_rag_config.chart_to_table_prompt

    chart_max_images = args.chart_caption_max_images
    if chart_max_images is not None and chart_max_images <= 0:
        chart_max_images = None

    parse_cache_dir: Optional[Path]
    if args.no_parse_cache:
        parse_cache_dir = None
    else:
        parse_cache_dir = args.parse_cache_dir if args.parse_cache_dir is not None else default_rag_config.parse_cache_dir

    debug_store_pages = default_rag_config.debug_store_pages if args.debug_store_pages is None else args.debug_store_pages
    debug_store_tables = default_rag_config.debug_store_tables if args.debug_store_tables is None else args.debug_store_tables
    debug_store_chart_tables = (
        default_rag_config.debug_store_chart_tables
        if args.debug_store_chart_tables is None
        else args.debug_store_chart_tables
    )
    debug_store_captions = default_rag_config.debug_store_captions if args.debug_store_captions is None else args.debug_store_captions
    debug_store_chunks = default_rag_config.debug_store_chunks if args.debug_store_chunks is None else args.debug_store_chunks
    debug_store_images = default_rag_config.debug_store_images if args.debug_store_images is None else args.debug_store_images

    debug_all = args.debug_all or None
    if debug_all:
        debug_store_pages = True
        debug_store_tables = True
        debug_store_chart_tables = True
        debug_store_captions = True
        debug_store_chunks = True
        debug_store_images = True

    rag_config = LangchainRAGConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        embedding_dimension=args.embedding_dimension,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_kwargs=embedding_kwargs,
        restrict_to_task_pages=not args.include_all_pages,
        page_padding=args.page_padding,
        use_mmr=args.use_mmr,
        mmr_lambda=args.mmr_lambda,
        mmr_fetch_k=args.mmr_fetch_k,
        max_context_characters=args.max_context_chars if args.max_context_chars > 0 else None,
        context_separator=args.context_separator,
        caption_charts=args.caption_charts,
        chart_caption_prompt=chart_prompt,
        chart_to_table_prompt=chart_to_table_prompt,
        chart_caption_max_images=chart_max_images,
        chart_caption_use_docling=not args.chart_caption_disable_docling,
        extract_tables=not args.disable_table_extraction,
        table_use_docling=not args.table_disable_docling,
        table_row_chunk_size=args.table_row_chunk_size,
        save_chunks_to_temp=args.save_chunks,
        chunk_dump_dir=args.chunk_dump_dir,
        parse_cache_dir=parse_cache_dir,
        debug_store_pages=debug_store_pages,
        debug_store_tables=debug_store_tables,
        debug_store_chart_tables=debug_store_chart_tables,
        debug_store_captions=debug_store_captions,
        debug_store_chunks=debug_store_chunks,
        debug_store_images=debug_store_images,
    )

    pipeline = LangchainRAGPipeline(
        benchmark_repo=benchmark_repo,
        model=model,
        artifacts_dir=args.artifacts,
        default_config=rag_config,
    )

    results = pipeline.run(
        experiment,
        rag_config=rag_config,
        experiment_suffix=args.experiment_suffix,
    )

    print("LangChain RAG run completed. Summary:")
    for row in [result.to_table_row() for result in results]:
        print(
            f"- {row['Task']}: {row['Model Output']} (expected: {row['Expected']}) | benchmark {row['Benchmark']}"
        )


if __name__ == "__main__":
    main()
