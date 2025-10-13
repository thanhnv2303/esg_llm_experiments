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

1. **Document loading** – The pipeline uses `PyPDFLoader` to read the report and, by default, limits indexing to the pages referenced by the experiment tasks (with optional padding). This keeps the vector store compact.
2. **Chunking and embeddings** – Chunks are produced with `RecursiveCharacterTextSplitter`. Embeddings rely on a deterministic hashing function (`HashingEmbeddings`) so no external models or downloads are required, making the pipeline portable.
3. **Retrieval** – A FAISS vector store powers either similarity or maximal marginal relevance (MMR) search. Retrieved snippets are normalised, previewed, and stored for inspection.
4. **Prompt construction** – Each task reuses the existing benchmark-aware prompt template and appends the retrieved context. The selected model backend (dummy, OpenAI-compatible, Groq, Google, Cerebras, or RAG-Anything) receives the enriched prompt.
5. **Outputs** – For every task, the pipeline persists the composed prompt, retrieved context (`contexts/{task_id}.md`), raw model responses, and a JSON payload capturing retrieval metadata. A summary table mirrors the structure produced by the original pipeline for easy comparison.

## CLI usage

Run the helper script from the repository root. The most important options are:

- `--chunk-size` / `--chunk-overlap` control the text splitter used before indexing.
- `--top-k` sets how many chunks are supplied to the prompt; combine with `--use-mmr`, `--mmr-lambda`, and `--mmr-fetch-k` to switch retrieval mode.
- `--include-all-pages` and `--page-padding` toggle which report pages are indexed.
- `--embedding-dimension`, `--max-context-chars`, and `--context-separator` adjust context formatting and size.
- `--model-backend` mirrors the primary CLI and accepts identical API configuration flags (API keys, retry parameters, headers, etc.).
- `--experiment-suffix` appends a readable tag to the derived artifacts directory so you can keep multiple runs for the same dataset/LLM side by side (e.g. `..._chunk900-overlap200-k4-sim-dim768-baseline`).

Padding applies per task. With `--page-padding 1`, a task tied to page 84 will also index pages 83 and 85 (clamped to the valid range). This keeps the retrieval window tight yet resilient when crucial figures or captions spill onto neighbouring pages. Leave padding at `0` for the strictest scope, or combine it with `--include-all-pages` when you want to embed the whole PDF regardless of task list.

The pipeline derives an identifier using `<company>_<year>_<model-name>_<retrieval-descriptor>`. When you pass `--experiment-suffix`, that string is appended at the end, giving you predictable artifact folders such as `prada_spa_2023_dummy_chunk900-overlap200-k4-sim-dim768-baseline`. Use different suffixes (e.g. `sim`, `mmr`, `gpt4o`) to make comparisons easier.

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

Because chunk previews are stored in Markdown, you can open them in any editor to quickly verify whether retrieval found the right region of the report. When experimenting with API-backed models, also monitor latency and token usage in the `responses/*.json` metadata section.

## Notes and best practices

- `HashingEmbeddings` is deterministic but lightweight; swap it for a richer embedding model by editing `LangchainRAGPipeline` if higher recall is needed.
- Because the pipeline reuses `ModelRunner`, any backend that works with the original pipeline will work here as well (API keys are still sourced from the environment or CLI flags).
- The context files are Markdown-formatted, making them easy to inspect or feed into downstream evaluation scripts.
- When using API-backed models, prefer setting keys through `.env` and the provided `--*-api-key-env` options so they are not stored in shell history.
- For large documents, consider enabling `--include-all-pages` only after testing page-restricted runs, as full indexing will increase embedding time and storage.
- If FAISS complains about missing AVX2 support, install the CPU-specific wheel shipped in `requirements.txt` or fall back to an alternative vector store supported by LangChain.

For additional implementation details, read `src/esg_pipeline/langchain_rag_pipeline.py`, which documents the retrieval flow end-to-end.
