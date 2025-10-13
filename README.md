# ESG Visual LLM Benchmark Harness


## Project Layout

```
.
├── data/
│   ├── NFD/                         # Source sustainability reports (PDF)
└── benchmarks/
    └── industry_indicators.csv      # Industry averages per indicator and year
├── experiments/
│   └── prada_2023.json              # Sample experiment definition
├── src/
│   └── esg_pipeline/
│       ├── benchmarks.py            # Benchmark repository loader
│       ├── config.py                # Experiment configuration dataclasses
│       ├── models/                  # Model interfaces (dummy + OpenAI, Google, Groq, Cerebras)
│       ├── pipeline.py              # Experiment runner
│       ├── preprocessing/           # PDF page extraction helpers
│       └── prompting.py             # Prompt template builder
└── main.py                          # CLI entry point
```

## Data Benchmarks

The table from `direction.md` is encoded in `data/benchmarks/industry_indicators.csv`. Each row contains:

- `indicator`: canonical indicator name
- `unit`: measurement unit (percent values are stored as decimals)
- `year`: benchmark year (2018–2024)
- `value`: numeric benchmark value

Helper functions in `src/esg_pipeline/benchmarks.py` expose these values to the prompt builder and evaluation pipeline.

## Running an Experiment

1. Ensure required Python packages and external tools are available:
   - Python 3.10+
   - Install dependencies with `pip install -r requirements.txt`
   - For the default pipeline, install [`PyMuPDF`](https://pymupdf.readthedocs.io/) or ensure Poppler utilities (`pdftoppm`/`pdftotext`) are on your `PATH`.
   - Optional: install [`docling`](https://github.com/ibm/docling) to enable Markdown-based extraction (`pip install docling`).
2. Activate your virtual environment if needed (not created automatically).
3. Execute the CLI with an experiment definition:

```bash
python main.py --experiment experiments/prada_2023.json
```

Key switches:
- `--model-backend {dummy,openai,google,groq,cerebras,rag-anything}` selects the model adapter (dummy returns a fixed label; OpenAI-compatible posts to a local/remote API; Google adapter sends requests to Gemini directly; Groq adapter calls Groq's OpenAI-compatible endpoint; Cerebras adapter targets Cerebras Inference; `rag-anything` runs the LightRAG-powered pipeline described below).
- `--openai-model`, `--api-base`, `--api-key/--api-key-env`, `--temperature`, `--top-p`, `--max-tokens`, `--request-timeout`, and `--extra-header` configure the OpenAI-compatible adapter.
- `--groq-model`, `--groq-api-base`, `--groq-api-key/--groq-api-key-env`, `--temperature`, `--top-p`, `--max-tokens`, `--request-timeout`, `--extra-header`, `--retry-attempts`, and `--retry-wait` configure the Groq adapter.
- `--cerebras-model`, `--cerebras-api-base`, `--cerebras-api-key/--cerebras-api-key-env`, `--temperature`, `--top-p`, `--max-tokens`, `--request-timeout`, `--extra-header`, `--retry-attempts`, and `--retry-wait` configure the Cerebras adapter.
- `--google-model`, `--google-api-base`, `--google-api-key/--google-api-key-env`, `--temperature`, `--top-p`, `--google-top-k`, `--google-max-output-tokens`, `--google-safety`, and `--request-timeout` configure the Google adapter.
- `--retry-attempts` and `--retry-wait` control how the CLI backs off and retries when the upstream API reports rate limits (HTTP 429).
- `--benchmarks PATH`, `--artifacts DIR`, and `--experiment-id NAME` adjust inputs and artifact locations.
- `--no-images` / `--no-text` skip respective preprocessing stages when the downstream model does not need them.
- `--resume` skips tasks that already have saved responses under the target artifacts directory, letting you continue an interrupted run.
- `--rag-config PATH` provides the JSON configuration used by the RAG-Anything backend. This flag is required when `--model-backend=rag-anything`.
- `--pdf-extractor {pymupdf,docling}` selects the PDF preprocessing backend. `docling` converts each page to Markdown and can optionally export detected figures/tables as separate images.
- `--docling-image-mode {embedded,referenced}` controls how Docling handles images: `embedded` inlines them as base64 inside the Markdown (no page screenshots are generated), while `referenced` saves crops to `images/<task>_assets/`, renders a combined figure with path captions, and sends that single composite image to the model.

When running with `--pdf-extractor=docling --docling-image-mode=referenced`, the pipeline saves the Markdown page under `texts/` (with a `.md` extension) and the associated crops under `images/<task>_assets/`. These additional images are supplied to vision-capable models alongside the primary page screenshot.

Set your API key via `--api-key`, `--groq-api-key`, `--cerebras-api-key`, or the corresponding environment variables (defaults `OPENAI_API_KEY`, `GROQ_API_KEY`, and `CEREBRAS_API_KEY`). Additional HTTP headers for self-hosted gateways can be supplied with repeated `--extra-header KEY=VALUE` arguments.

*Note*: Groq and Cerebras chat completion endpoints are currently text-only. The adapters therefore ignore extracted page images and rely on the OCR text produced during preprocessing.

**Groq example**

```bash
export GROQ_API_KEY=your-groq-key

python main.py --experiment experiments/prada_2023.json \
  --model-backend groq \
  --groq-model llama-3.2-11b-vision-preview
```

**Cerebras example**

```bash
export CEREBRAS_API_KEY=your-cerebras-key

python main.py --experiment experiments/prada_2023.json \
  --model-backend cerebras \
  --cerebras-model llama3.1-8b
```

Interrupted runs can be continued safely with `--resume`, which reuses the existing artifacts for tasks that already produced responses and only evaluates the remaining ones.

## Outputs

Running the CLI creates an `artifacts/<experiment-id>/` directory containing:

- `prompts/` – prompt text sent to the model for each task
- `images/` – extracted page images (if enabled)
- `texts/` – extracted page text (if enabled)
- `responses/` – raw and structured model responses
- `summary.json` – quick comparison table (`page`, `indicator`, `benchmark`, `model output`, `expected`)

These artifacts provide the audit trail necessary to compare multiple models over the same PDF pages.

## Extending the Framework

1. Duplicate `experiments/prada_2023.json` and adjust company, pages, indicators, and expected outcomes.
2. Add more benchmark values to the CSV if you introduce new indicators.
3. Implement real model runners (e.g., Qwen, GPT-4o) that call local weights or API endpoints and return `ModelResponse` objects.
4. Run the CLI multiple times, each with a different model, to gather comparable summaries.

This structure transforms the guidance in `direction.md` into an experiment harness that scales to multiple companies, indicators, and models.

### RAG-Anything Pipeline

The `rag-anything` backend wraps the [RAG-Anything](https://github.com/HKUDS/RAG-Anything/tree/main) workflow so you can combine LightRAG retrieval with the existing ESG benchmark tasks. To enable it:

1. Create a RAG configuration JSON (see `configs/rag_anything_example.json`). At minimum specify the LLM endpoint under `llm` and the embedding model under `embedding`. Optional sections let you customise the parser/working directory (`rag_anything`), LightRAG storage backends (`lightrag`), query options, and an optional vision-specific endpoint.
2. Provide credentials either inline via `api_key`, by pointing `api_key_file` at a secrets file, or by referencing an environment variable with `api_key_env`.
3. Run the CLI with `--model-backend=rag-anything --rag-config configs/rag_anything_example.json` plus your usual `--experiment` and artifact arguments.

The runner invokes `process_document_complete` once per report to populate the LightRAG stores, then issues structured queries for each task. When `query.use_multimodal` is true and page images are available the pipeline automatically forwards them via the RAG-Anything multimodal API.

Config structure (`configs/rag_anything_example.json`):
- `rag_anything` controls how RAG-Anything parses reports (set the LightRAG working directory with `working_dir`, choose the document parser via `parser`, decide the default `parse_method`, and toggle multimodal asset extraction with the `enable_*` flags).
- `llm` defines the text-generation endpoint that powers both retrieval reasoning and captions; specify `provider`/`model`, supply credentials with `api_key`, `api_key_file`, or `api_key_env`, and pass sampling limits (`temperature`, `top_p`, `max_tokens`) plus optional OpenAI-compatible `api_base`.
- `embedding` configures the vector store encoder (`model`, `embedding_dim`, `max_token_size`) and typically reuses the same API key scope; you can reuse the LLM key via `api_key`/`api_key_file` or fall back to `api_key_env`, and set `extra_params.input_type` (`passage` or `query`) plus optional `extra_params.extra_body` for provider-specific fields such as NVIDIA's `truncate`.
- `vision_model` is optional and overrides the LLM with a multimodal-capable model for image captions; omit it to fall back to the plain `llm` settings.
- `lightrag` forwards extra keyword arguments to `LightRAG`, for example storage backends (`vector_storage`, `kv_storage`, `graph_storage`) or chunk sizes; adjust these to match local infrastructure.
- `document_processing` tunes the one-off ingestion pass: choose an alternative `parse_method`, disable CLI statistics, set a default inference `device` (e.g., "cuda" to force GPU parsing), optionally enforce it via `device_enforce`, declare a fallback `device_virtual_memory_gb` to skip MinerU’s VRAM probe, or forward parser-specific options via `parser_kwargs`. Additional fields let you constrain ingestion to the pages you actually query: set `restrict_to_experiment_pages` to use the experiment task list, widen the window with `experiment_page_padding`, add explicit intervals via `page_ranges`/`extra_pages`, and enable `clear_existing_document` to purge any prior LightRAG state when reusing a `doc_id`. Docling backends use PyMuPDF to slice the PDF when a page window is requested and run in verbose mode by default; override with `DOCLING_VERBOSE_LEVEL`/`DOCLING_VERBOSE` (set to `0`, `1`, or `2`) if you prefer quieter or noisier logs.
- `query` governs per-task retrieval: pick the LightRAG query `mode`, decide whether to send images with `use_multimodal`, and capture default kwargs such as decoding `temperature` inside `params`.

> **Security note**: storing API keys directly in the JSON makes the file sensitive. Keep it out of version control or use `api_key_file`/`api_key_env` when that suits your workflow better.

**Example run script**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Point these at your actual keys or swap to api_key_file/api_key_env in the JSON
export OPENAI_API_KEY=your-openai-key

python main.py \
  --experiment experiments/prada_2023.json \
  --model-backend rag-anything \
  --rag-config configs/rag_anything_example.json \
  --artifacts artifacts/rag_anything_demo
```

### LangChain RAG pipeline

If you prefer a streamlined LangChain-based RAG flow (with lightweight hashing embeddings and FAISS retrieval), use `scripts/run_langchain_rag.py`. An end-to-end explanation, setup checklist, and usage examples live in `docs/langchain_rag_pipeline.md`.

Example dummy run:

```bash
source .venv/bin/activate
python scripts/run_langchain_rag.py \
  --experiment experiments/prada_2023.json \
  --model-backend dummy \
  --chunk-size 900 \
  --top-k 4 \
  --experiment-suffix baseline
```

The script mirrors the primary CLI’s model configuration flags while exposing additional retrieval parameters (`--chunk-size`, `--top-k`, `--use-mmr`, etc.). See the documentation for more guidance and troubleshooting tips.

## Docker Compose Setups

### Local Ollama service

A ready-to-run Compose file (`docker-compose.yml`) starts an Ollama server on `localhost:11434`. The container automatically pulls the model specified in `OLLAMA_DEFAULT_MODEL` (defaults to `llama3.1:8b`) and stores weights in a local Docker volume.

```bash
# Start Ollama with the default model
OLLAMA_DEFAULT_MODEL=llama3.1:8b docker compose up -d ollama

# Point the evaluation harness at the local endpoint
python main.py --experiment experiments/prada_2023.json \
  --model-backend openai \
  --api-base http://localhost:11434 \
  --openai-model llama3.1:8b \
  --api-key dummy-token
```

Set `OLLAMA_PULL_ONLY=true` when you only need to download weights without keeping the server running.

### Google Gemini (direct or via LiteLLM proxy)

You can call Gemini either through the built-in Google adapter or through the optional LiteLLM proxy.

**Direct adapter**

```bash
export GOOGLE_API_KEY=your-gemini-key

python main.py --experiment experiments/prada_2023.json \
  --model-backend google \
  --google-model gemini-2.5-flash
```

Adjust sampling behaviour with `--temperature`, `--top-p`, and `--google-top-k`, and limit generations with `--google-max-output-tokens` or the generic `--max-tokens` flag. Safety overrides accept repeated `--google-safety CATEGORY=THRESHOLD` values when you need custom thresholds.

Both the OpenAI-compatible and Google adapters automatically wait and retry when the API returns HTTP 429 rate-limit responses; tune this with `--retry-attempts` and `--retry-wait`.

**LiteLLM proxy**

LiteLLM (enabled with `--profile google`) wraps Gemini in an OpenAI-compatible interface if you prefer to re-use the OpenAI adapter.

```bash
export GOOGLE_API_KEY=your-gemini-key
docker compose --profile google up -d litellm

python main.py --experiment experiments/prada_2023.json \
  --model-backend openai \
  --api-base http://localhost:4000 \
  --openai-model gemini-2.5-flash
```

### RAG-Anything storage stack

For `rag-anything` runs that rely on external storage engines, a dedicated
Compose file provisions Neo4j (graph), Redis (key-value), Milvus (vector), and
MongoDB (document status).

```bash
# Start the storage services (wrapper around docker compose)
scripts/manage_rag_storage.sh up

# Export connection settings expected by LightRAG
source local/configs/rag_anything_storage.env

# Point the pipeline at the distributed storage config
python main.py --experiment experiments/prada_2023.json \
  --model-backend rag-anything \
  --rag-config local/configs/rag_anything_distributed.json

# Optionally stop the stack when done
scripts/manage_rag_storage.sh down
```

Run `scripts/manage_rag_storage.sh cleanup` when you want to tear everything
down and remove persisted volumes, or `scripts/manage_rag_storage.sh follow-logs`
to tail the stack in real time.

If you need to wipe all persisted RAG state between runs without tearing down
the containers, use the cleanup helper:

```bash
python scripts/cleanup_rag_storage.py             # destructive cleanup
python scripts/cleanup_rag_storage.py --dry-run   # preview only
```

Update `configs/litellm_config.yaml` to match the proxied model name or sampling parameters when needed.
`configs/rag_anything_docling_subset.json` shows a Docling-on-CUDA setup that enables LightRAG chunking, clears an existing document before ingestion, and restricts parsing to the experiment’s task pages plus a configurable buffer.
