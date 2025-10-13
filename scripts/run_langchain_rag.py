#!/usr/bin/env python3
"""CLI utility to run ESG experiments with the LangChain RAG pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

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

    rag_config = LangchainRAGConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        embedding_dimension=args.embedding_dimension,
        restrict_to_task_pages=not args.include_all_pages,
        page_padding=args.page_padding,
        use_mmr=args.use_mmr,
        mmr_lambda=args.mmr_lambda,
        mmr_fetch_k=args.mmr_fetch_k,
        max_context_characters=args.max_context_chars if args.max_context_chars > 0 else None,
        context_separator=args.context_separator,
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
