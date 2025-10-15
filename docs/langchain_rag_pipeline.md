# LangChain RAG Pipeline

This pipeline augments the existing ESG evaluation workflow with retrieval-augmented generation built on top of LangChain. It wraps the familiar experiment configuration (`experiments/*.json`) and model backends while replacing the per-page extraction logic with chunked retrieval across the report. Artifacts are written under the same `artifacts/` directory so you can compare runs easily.

At a high level you:

1. Install the LangChain/FAISS dependencies (already listed in `requirements.txt`).
2. Point the helper script at an experiment JSON and a model backend.
3. Review the generated prompts, contexts, and responses inside the derived artifacts folder.

The sections below walk through the details.

## Prerequisites

- Python 3.10+ (matches the project interpreter).
- The project virtual environment activated: `source .venv/bin/activate`.
- Required packages installed: `pip install -r requirements.txt`.
- Access to the experiment PDF referenced by the JSON configuration (e.g. `data/NFD/PRADA S.P.A/NFD2023.pdf`).
- For non-dummy backends, the relevant API keys exported in your environment or stored in `.env` (OpenAI-compatible, Groq, Google, Cerebras, etc.).

## Quick start

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_langchain_rag.py \
  --experiment experiments/prada_2023.json \
  --model-backend dummy \
  --chunk-size 900 \
  --top-k 4 \
  --experiment-suffix dummy-baseline
```

This command:

- loads the PRADA experiment definition;
- builds a LangChain vector store restricted to the relevant PDF pages;
- runs the dummy model (useful for validating retrieval without LLM calls);
- stores outputs under `artifacts/prada_spa_2023_dummy_chunk900-overlap200-k4-sim-dim768-dummy-baseline/`.

## How it works

1. **Page parsing** – `PyPDFLoader` extracts the textual layer for every requested page, while Docling (when available) generates Markdown snapshots, table structures, and referenced chart images. Parsed artefacts are cached on disk (`--parse-cache-dir`) so reruns avoid the expensive PDF conversion step.
2. **Chart understanding** – Each Docling-exported figure is OCR'd (via `pytesseract`) to ensure it contains readable text or numbers; empty figures are skipped early. Remaining candidates are sent to the configured chart model, which responds in JSON, confirming the asset is a chart and emitting a Markdown table plus caption in a single pass. Each successful conversion becomes its own table chunk and is referenced from the originating page so retrieval can recover it together with the original image path.
3. **Table enrichment** – Native tables and chart-derived tables stay in Markdown format. Short tables are embedded as a single chunk; longer ones are split row-wise while repeating headers so structure is preserved. Metadata keeps track of origin (document vs chart), captions, and context text.
4. **Text chunking & embeddings** – Non-tabular page text is chunked with `RecursiveCharacterTextSplitter` using the configured size/overlap. Table documents are flagged as pre-chunked so they are ingested without modification, and each chunk carries the full Markdown table so downstream consumers can reconstruct the entire grid. Embeddings are created via `create_embeddings`, honouring `LangchainRAGConfig.embedding_backend` to swap between hashing, HuggingFace, or OpenAI vectors without code changes.
5. **Retrieval & prompting** – A FAISS vector store powers similarity or MMR search. Retrieved snippets are formatted into Markdown, appended to the benchmark-aware prompt, and sent to the chosen model backend (dummy, OpenAI-compatible, Groq, Google, Cerebras, or RAG-Anything).
6. **Outputs & debugging** – For each task the pipeline stores the composed prompt, retrieved context (`contexts/{task_id}.md`), raw model responses, and JSON metadata (including chart tables and skipped figures). Optional debug artefacts (page Markdown, extracted tables, chart tables, raw vision responses) are written under `debug/` when the corresponding flags are enabled, alongside the consolidated `chart_table_summary.json`.

## CLI usage

Run the helper script from the repository root. The most important options are:

- `--chunk-size` / `--chunk-overlap` control the text splitter used before indexing.
- `--top-k` sets how many chunks are supplied to the prompt; combine with `--use-mmr`, `--mmr-lambda`, and `--mmr-fetch-k` to switch retrieval mode.
- `--include-all-pages` and `--page-padding` toggle which report pages are indexed.
- `--embedding-backend`, `--embedding-model`, and repeated `--embedding-option key=value` let you pick and configure the embedder; fall back to `--embedding-dimension` only matters for the hashing backend.
- `--max-context-chars` and `--context-separator` adjust how retrieved snippets are formatted before prompting the model.
- `--caption-charts` / `--no-caption-charts` toggle the chart-to-table conversion stage. Fine tune the conversion with `--chart-to-table-prompt`, the legacy `--chart-caption-prompt`, and `--chart-caption-max-images`.
- `--parse-cache-dir` (plus `--no-parse-cache`) governs where Docling outputs and page Markdown are cached. Leave the default to avoid reprocessing large PDFs on reruns.
- `--save-chunks` stores the post-processed chunks in a temporary directory; pair with `--chunk-dump-dir` to choose a persistent location.
- `--disable-table-extraction`, `--table-row-chunk-size`, and `--table-disable-docling` control how tabular data is linearised and whether Docling is used.
- `--debug-store-pages`, `--debug-store-tables`, `--debug-store-chart-tables`, and `--debug-store-captions` materialise the intermediate Markdown artefacts under `debug/` for inspection.
- `--model-backend` mirrors the primary CLI and accepts identical API configuration flags (API keys, retry parameters, headers, etc.).
- `--experiment-suffix` appends a readable tag to the derived artifacts directory so you can keep multiple runs for the same dataset/LLM side by side (e.g. `..._chunk900-overlap200-k4-sim-dim768-baseline`).

Padding applies per task. With `--page-padding 1`, a task tied to page 84 will also index pages 83 and 85 (clamped to the valid range). This keeps the retrieval window tight yet resilient when crucial figures or captions spill onto neighbouring pages. Leave padding at `0` for the strictest scope, or combine it with `--include-all-pages` when you want to embed the whole PDF regardless of task list.

The pipeline derives an identifier using `<company>_<year>_<model-name>_<retrieval-descriptor>`. When you pass `--experiment-suffix`, that string is appended at the end, giving you predictable artifact folders such as `prada_spa_2023_dummy_chunk900-overlap200-k4-sim-dim768-baseline`. Use different suffixes (e.g. `sim`, `mmr`, `gpt4o`) to make comparisons easier.

### Embedding backends

`LangchainRAGConfig` now exposes `embedding_backend`, `embedding_model`, and `embedding_kwargs`, so you can swap vectorisers without modifying pipeline code. The helper CLI mirrors these controls through `--embedding-backend`, `--embedding-model`, and repeated `--embedding-option key=value` flags.

| Backend value | Default model | Extra requirement | Typical options |
| --- | --- | --- | --- |
| `hash` / `hashing` | Deterministic hashing (no external model) | none | `--embedding-dimension 1536` to adjust vector size |
| `huggingface`, `hf`, `sentence-transformers` | `sentence-transformers/all-mpnet-base-v2` | `pip install sentence-transformers` | `--embedding-model sentence-transformers/all-MiniLM-L12-v2`, `--embedding-option cache_folder=.hf-cache` |
| `openai`, `openai-embeddings`, `azure-openai` | `text-embedding-3-small` | `pip install langchain-openai` + API key | `--embedding-model text-embedding-3-large`, `--embedding-option api_key=...` when not set via env |

Example using a HuggingFace encoder:

```bash
python scripts/run_langchain_rag.py \
  --experiment experiments/prada_2023.json \
  --model-backend groq \
  --groq-model llama-3.2-11b-vision-preview \
  --embedding-backend huggingface \
  --embedding-model sentence-transformers/all-MiniLM-L12-v2 \
  --embedding-option cache_folder=.hf-cache \
  --experiment-suffix hf-minilm
```

Programmatic usage mirrors the CLI:

```python
from esg_pipeline import LangchainRAGConfig

config = LangchainRAGConfig(
    embedding_backend="openai",
    embedding_model="text-embedding-3-large",
    embedding_kwargs={"api_key": "<your-key>", "dimensions": 3072},
)
```

### Chart-aware context

Passing `--caption-charts` enables the chart-to-table conversion stage: page snapshots feed Docling-isolated figures to the configured vision model, which must return JSON describing whether the image is a chart and, if so, a Markdown table with a caption. The converted tables are attached to the originating page, indexed as retrieval chunks, and referenced in the response payload alongside the original image path.

- Override the conversion instructions with `--chart-to-table-prompt` (or the legacy `--chart-caption-prompt`) to enforce a specific JSON/table schema.
- Limit images per page with `--chart-caption-max-images N` (use `0` or a negative number for "all").
- Set `--chart-caption-disable-docling` if you cannot install Docling; the pipeline falls back to whole-page raster images, reducing the precision of figure isolation.

Because the conversion request routes through `ModelRunner`, choose a backend that accepts images (e.g. Groq vision models or OpenAI `gpt-4o`). The same model is reused for chart analysis by default; instantiate `LangchainRAGPipeline` with a dedicated `chart_model` if you need a specialised ESG vision checkpoint. The resulting chart tables are recorded in `responses/{task_id}.json` under `chart_insights` together with optional `skipped_charts` entries explaining why images were ignored.

Whenever retrieval surfaces a table chunk (either native or chart-derived), the formatter injects the full Markdown table into the context so the answering model always sees the complete grid alongside the surrounding prose.

### Retrieval parameter reference

| Flag | Default | Description |
| --- | --- | --- |
| `--chunk-size` | 1000 | Target number of characters per chunk passed to the embedder. Larger chunks keep more context but may dilute embedding focus; smaller chunks improve recall at the cost of more vectors. |
| `--chunk-overlap` | 200 | Number of characters to overlap between consecutive chunks. Increases recall for concepts straddling chunk boundaries. Set close to `chunk-size` to mimic sliding windows; reduce for faster indexing. |
| `--embedding-dimension` | 768 | Size of the hashing-based embedding vector. Higher values improve separation but increase FAISS memory. Stick to powers of two when possible. |
| `--top-k` | 4 | Number of retrieved chunks injected into the prompt. Raising `k` increases context breadth and prompt length; lowering keeps responses focused. |
| `--use-mmr` | `False` | Enables maximal marginal relevance retrieval. When true, the vector store balances similarity with diversity using the parameters below. Leave disabled for pure similarity search. |
| `--mmr-lambda` | 0.5 | Trade-off between relevance and novelty during MMR search. Values near 1 favour similarity (results similar to the query); values near 0 emphasise diversity across chunks. |
| `--mmr-fetch-k` | 12 | Pool size of candidate chunks considered when MMR is active. Needs to be ≥ `top-k`. Increase if the document is dense and you want more variety during selection. |
| `--include-all-pages` | `False` | When set, indexes every PDF page instead of only the task-specific pages (plus padding). Useful for ad hoc explorations; expect longer preprocessing time. |
| `--page-padding` | 0 | Expands the indexed window around each task page. With padding `p`, a task on page `N` includes pages `N-p`…`N+p` (within bounds). Keeps retrieval local while tolerating neighbouring figures. |
| `--max-context-chars` | 4000 | Caps the concatenated Markdown context length. Set to `0` or a negative value to disable truncation. Helps prevent runaway prompts with high `top-k`. |
| `--context-separator` | `\n\n---\n\n` | String inserted between retrieved chunks in the Markdown context file. Adjust if your model prefers different delimiters. |
| `--experiment-suffix` | `None` | Optional tag appended to the derived artifacts directory so multiple runs remain distinguishable. |
| `--log-level` | `INFO` | Standard Python logging level the script configures. Switch to `DEBUG` for detailed retrieval diagnostics. |
| `--caption-charts` | `True` | Enables chart detection and conversion. Pair with `--no-caption-charts` to skip the vision step entirely. |
| `--chart-to-table-prompt` | JSON instruction | Vision prompt that asks the model to confirm the asset is a chart and emit a Markdown table + caption. Override to customise formatting or languages. |
| `--chart-caption-max-images` | `-1` | Maximum number of images per page to process (≤0 means "all"). Useful when slides contain dozens of small charts. |
| `--chart-caption-disable-docling` | `False` | Skip Docling and fall back to full-page raster images. Expect lower chart detection accuracy because individual figures cannot be isolated. |
| `--parse-cache-dir` | `.rag_cache/` | Directory where parsed page Markdown, tables, and images are cached. Delete or use `--no-parse-cache` when you need a clean re-run. |
| `--debug-store-pages` | `False` | Write per-page Markdown renderings (text + tables + chart tables) under `artifacts/<run>/debug/pages/`. |
| `--debug-store-tables` | `True` | Persist every extracted table chunk as Markdown under `debug/tables/` for inspection. |
| `--debug-store-chart-tables` | `True` | Persist the Markdown tables generated from charts under `debug/charts/`. |
| `--debug-store-captions` | `False` | Store the raw JSON/LLM responses from the vision model under `debug/charts/` to aid troubleshooting. |
| `--debug-store-chunks` | `True` | Persist the final retrieved chunks (Markdown + metadata) under `debug/chunks/` for auditing the context actually sent to the model. |
| `--debug-store-images` | `True` | Copy Docling-extracted chart/page images into `debug/images/` so you can inspect the visuals used during conversion. |

### Supplying API credentials

- **OpenAI-compatible**: use `--api-key` or set `OPENAI_API_KEY` / a custom `--api-key-env`.
- **Groq**: use `--groq-api-key` or set `GROQ_API_KEY`.
- **Google**: use `--google-api-key` or set `GOOGLE_API_KEY`; optional safety settings via `--google-safety CATEGORY=THRESHOLD`.
- **Cerebras**: use `--cerebras-api-key` or set `CEREBRAS_API_KEY`.
- **RAG-Anything**: provide `--rag-config` pointing to an existing pipeline configuration JSON.

Credentials can also be placed in `.env`; the script calls `dotenv.load_dotenv` so they are picked up automatically.

### Overriding retrieval defaults

All retrieval parameters have CLI flags, so you can create shell scripts or VS Code tasks for commonly used presets. Example with MMR and additional padding:

```bash
python scripts/run_langchain_rag.py \
  --experiment experiments/prada_2023.json \
  --model-backend openai \
  --openai-model gpt-4o-mini \
  --chunk-size 1100 \
  --chunk-overlap 250 \
  --top-k 6 \
  --use-mmr \
  --mmr-lambda 0.35 \
  --page-padding 1 \
  --experiment-suffix gpt4o-mmr
```

## Running parameter sweeps

To contrast configurations, launch the script multiple times while varying retrieval parameters. For example:

```bash
# Focused context with strict similarity search
a=.venv/bin/python scripts/run_langchain_rag.py \
  --experiment experiments/prada_2023.json \
  --model-backend dummy \
  --chunk-size 800 \
  --top-k 3 \
  --experiment-suffix sim-k3

# Broader context using MMR with larger chunks
a=.venv/bin/python scripts/run_langchain_rag.py \
  --experiment experiments/prada_2023.json \
  --model-backend dummy \
  --chunk-size 1200 \
  --chunk-overlap 250 \
  --top-k 6 \
  --use-mmr \
  --mmr-lambda 0.4 \
  --experiment-suffix mmr-k6
```

Each run produces a distinct artifacts folder (suffix appended to the derived experiment id). Compare `contexts/` and `responses/` across runs to gauge how retrieval choices affect the prompts and model outputs.

## Inspecting outputs

Inside each artifacts directory you will find:

- `contexts/{task_id}.md` – Markdown files with retrieved snippets, chunk headers, and optional similarity scores.
- `prompts/{task_id}.txt` – the final prompt delivered to the model (benchmark instructions + context).
- `responses/{task_id}.json` – structured payload containing the model reply, metadata, query, and retrieved chunk summaries.
- `responses/{task_id}.txt` – raw text response for quick inspection.
- `summary.json` – list of tasks with predicted labels, expected labels, and benchmark values, matching the format produced by the traditional pipeline.
- `charts/page_XXXX.md` – page-level ESG commentary generated prior to chunking; this is the exact text injected into the retrieval corpus.
- `charts/{task_id}/` – when chart captioning is enabled, contains copies of the analysed figures for that task (filenames include the `chart_id` for easy traceability).
- `chunk_{index}.json` files under the path reported in the logs when `--save-chunks` or `--chunk-dump-dir` is used; each JSON holds the chunk text and metadata as stored in the vector store.
- Tabular chunks are labelled with `[table:...]` markers and include the original header in every slice; use their metadata (`table_id`, `table_caption`, `table_context`) to trace responses back to the PDF tables.
- `debug/` – a companion folder containing `run_config.json`, per-page Markdown renders, full table/chunk Markdown exports, chart-to-table conversions, raw vision responses, and copies of the Docling-extracted images (all toggleable via the `--debug-store-*` flags).

Because chunk previews are stored in Markdown, you can open them in any editor to quickly verify whether retrieval found the right region of the report. When experimenting with API-backed models, also monitor latency and token usage in the `responses/*.json` metadata section.

## Notes and best practices

- `--embedding-backend` lets you trade hashing for stronger embeddings without code changes; `hash` is great for reproducibility, while HuggingFace and OpenAI options improve semantic recall.
- Because the pipeline reuses `ModelRunner`, any backend that works with the original pipeline will work here as well (API keys are still sourced from the environment or CLI flags).
- The context files are Markdown-formatted, making them easy to inspect or feed into downstream evaluation scripts.
- When using API-backed models, prefer setting keys through `.env` and the provided `--*-api-key-env` options so they are not stored in shell history.
- For large documents, consider enabling `--include-all-pages` only after testing page-restricted runs, as full indexing will increase embedding time and storage.
- If FAISS complains about missing AVX2 support, install the CPU-specific wheel shipped in `requirements.txt` or fall back to an alternative vector store supported by LangChain.
- Chart-to-table conversion works best with Docling (`pip install docling pillow`) and a vision-capable backend. Inspect the generated tables under `debug/charts/` (enable with `--debug-store-chart-tables`) to ensure the structure matches the original figure.
- Enabling `--save-chunks` is helpful when you want to diff chunking strategies; remember to clean up the generated directory once you're done inspecting it.
- Install `langchain-huggingface` if you plan to use HuggingFace embeddings (`pip install langchain-huggingface`).
- Table extraction relies on Docling for structural parsing. For stubborn PDFs, try lowering `--table-row-chunk-size` to narrow each chunk and improve recall.
- Leave the parse cache enabled for iterative experiments; the first run populates `.rag_cache/`, while subsequent runs reuse the Markdown/tables without rerunning Docling. Remove the cache if the source PDF changes.
- OCR-based chart filtering depends on `pytesseract` (and the system `tesseract` binary) plus `Pillow`. Install them if you want the pipeline to skip decorative figures automatically; otherwise it falls back to processing every extracted image.

For additional implementation details, read `src/esg_pipeline/langchain_rag_pipeline.py`, which documents the retrieval flow end-to-end.
