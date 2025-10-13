from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)

from esg_pipeline import (  # noqa: E402
    BenchmarkRepository,
    CerebrasModel,
    DummyModel,
    ExperimentRunner,
    GoogleGenerativeModel,
    GroqModel,
    OpenAICompatibleModel,
    RAGAnythingModel,
    RAGAnythingPipelineConfig,
    load_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ESG visual LLM experiments.")
    parser.add_argument(
        "--experiment",
        required=True,
        type=Path,
        help="Path to experiment configuration file (JSON).",
    )
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=Path("benchmarks/industry_indicators.csv"),
        help="CSV file containing industry benchmark values.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Directory where prompts, responses, and summaries will be saved.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Optional identifier to use for the artifacts directory.",
    )
    parser.add_argument(
        "--model-backend",
        choices=["dummy", "openai", "google", "groq", "cerebras", "rag-anything"],
        default="dummy",
        help="Model backend to use when running the experiment.",
    )
    parser.add_argument(
        "--rag-config",
        type=Path,
        default=None,
        help="Path to a RAG-Anything configuration file (JSON). Required when --model-backend=rag-anything.",
    )
    parser.add_argument(
        "--dummy-label",
        type=str,
        choices=["higher", "lower", "equal", "not found"],
        default="not found",
        help="Prediction label returned by the dummy model.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="Model name supplied to the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com"),
        help="Base URL for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the OpenAI-compatible endpoint (overrides environment variable).",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name to read when --api-key is not provided.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the selected model backend.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling (top_p) for the selected model backend.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate when supported by the backend.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=180,
        help="Timeout in seconds for OpenAI-compatible HTTP requests.",
    )
    parser.add_argument(
        "--extra-header",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Extra HTTP header for the model endpoint; can be provided multiple times.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Number of retry attempts when the model endpoint returns rate-limit errors.",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=60,
        help="Base wait time in seconds before retrying after a rate-limit response.",
    )
    parser.add_argument(
        "--groq-model",
        type=str,
        default="llama-3.2-11b-vision-preview",
        help="Model name supplied to the Groq endpoint.",
    )
    parser.add_argument(
        "--groq-api-base",
        type=str,
        default=os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai"),
        help="Base URL for the Groq endpoint.",
    )
    parser.add_argument(
        "--groq-api-key",
        type=str,
        default=None,
        help="API key for the Groq endpoint (overrides environment variable).",
    )
    parser.add_argument(
        "--groq-api-key-env",
        type=str,
        default="GROQ_API_KEY",
        help="Environment variable name to read when --groq-api-key is not provided.",
    )
    parser.add_argument(
        "--cerebras-model",
        type=str,
        default="llama3.1-8b",
        help="Model name supplied to the Cerebras endpoint.",
    )
    parser.add_argument(
        "--cerebras-api-base",
        type=str,
        default=os.environ.get("CEREBRAS_API_BASE", "https://api.cerebras.ai"),
        help="Base URL for the Cerebras endpoint.",
    )
    parser.add_argument(
        "--cerebras-api-key",
        type=str,
        default=None,
        help="API key for the Cerebras endpoint (overrides environment variable).",
    )
    parser.add_argument(
        "--cerebras-api-key-env",
        type=str,
        default="CEREBRAS_API_KEY",
        help="Environment variable name to read when --cerebras-api-key is not provided.",
    )
    parser.add_argument(
        "--google-model",
        type=str,
        default="gemini-2.5-flash",
        help="Model name supplied to the Google Generative AI endpoint.",
    )
    parser.add_argument(
        "--google-api-base",
        type=str,
        default=os.environ.get(
            "GOOGLE_API_BASE", "https://generativelanguage.googleapis.com"
        ),
        help="Base URL for the Google Generative AI endpoint.",
    )
    parser.add_argument(
        "--google-api-key",
        type=str,
        default=None,
        help="API key for the Google Generative AI endpoint (overrides environment variable).",
    )
    parser.add_argument(
        "--google-api-key-env",
        type=str,
        default="GOOGLE_API_KEY",
        help="Environment variable name to read when --google-api-key is not provided.",
    )
    parser.add_argument(
        "--google-top-k",
        type=int,
        default=None,
        help="Top-K sampling parameter for Google Generative AI requests.",
    )
    parser.add_argument(
        "--google-max-output-tokens",
        type=int,
        default=None,
        help="Maximum number of output tokens for Google Generative AI requests.",
    )
    parser.add_argument(
        "--google-safety",
        action="append",
        default=None,
        metavar="CATEGORY=THRESHOLD",
        help="Safety setting override for Google Generative AI (can be repeated).",
    )
    parser.add_argument(
        "--pdf-extractor",
        choices=["pymupdf", "docling"],
        default="docling",
        help=(
            "PDF extraction backend. 'pymupdf' keeps the existing PyMuPDF/Poppler pipeline. "
            "'docling' converts the page to Markdown using Docling."
        ),
    )
    parser.add_argument(
        "--docling-image-mode",
        choices=["embedded", "referenced"],
        default="referenced",
        help=(
            "When using --pdf-extractor=docling, choose whether images are embedded as data URIs "
            "in the Markdown or exported as separate files referenced from the Markdown."
        ),
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip page image extraction during preprocessing.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Skip page text extraction during preprocessing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run by skipping tasks with existing responses.",
    )
    args = parser.parse_args()

    if args.pdf_extractor == "docling" and args.no_text:
        parser.error("Docling extractor requires page text capture. Remove --no-text to continue.")

    return args


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


def build_model(args: argparse.Namespace):
    if args.model_backend == "dummy":
        return DummyModel(default_label=args.dummy_label)

    if args.model_backend == "openai":
        api_key = args.api_key or os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI-compatible model requires an API key. Pass --api-key or set the environment variable."
            )
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
        api_key = (
            args.groq_api_key
            or os.environ.get(args.groq_api_key_env)
            or os.environ.get("GROQ_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Groq model requires an API key. Pass --groq-api-key or set the environment variable."
            )
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
            raise ValueError(
                "Google Generative AI model requires an API key. Pass --google-api-key or set the environment variable."
            )
        headers = parse_extra_headers(args.extra_header)
        safety_settings = parse_google_safety_settings(args.google_safety)
        max_output_tokens = args.google_max_output_tokens
        if max_output_tokens is None:
            max_output_tokens = args.max_tokens
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

    if args.model_backend == "rag-anything":
        if args.rag_config is None:
            raise ValueError(
                "RAG-Anything backend requires --rag-config pointing to the pipeline configuration JSON."
            )
        rag_config = RAGAnythingPipelineConfig.from_json(args.rag_config)
        return RAGAnythingModel(config=rag_config, artifacts_dir=args.artifacts)

    if args.model_backend == "cerebras":
        api_key = (
            args.cerebras_api_key
            or os.environ.get(args.cerebras_api_key_env)
            or os.environ.get("CEREBRAS_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Cerebras model requires an API key. Pass --cerebras-api-key or set the environment variable."
            )
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

    raise ValueError(f"Unsupported model backend: {args.model_backend}")


def main() -> None:
    args = parse_args()

    benchmark_repo = BenchmarkRepository(args.benchmarks)
    experiment = load_experiment(args.experiment)

    model = build_model(args)

    if hasattr(model, "prepare_dataset"):
        model.prepare_dataset(experiment)

    runner = ExperimentRunner(
        benchmark_repo=benchmark_repo,
        model=model,
        artifacts_dir=args.artifacts,
        capture_images=not args.no_images,
        capture_text=not args.no_text,
        pdf_extractor=args.pdf_extractor,
        docling_image_mode=args.docling_image_mode,
    )

    results = runner.run(
        experiment,
        experiment_id=args.experiment_id,
        resume=args.resume,
    )

    print("Experiment completed. Summary:")
    for row in [result.to_table_row() for result in results]:
        print(
            f"- {row['Task']}: {row['Model Output']} (expected: {row['Expected']}) | "
            f"benchmark {row['Benchmark']}"
        )


if __name__ == "__main__":
    main()
