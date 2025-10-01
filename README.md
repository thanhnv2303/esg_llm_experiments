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
   - **Either** [`PyMuPDF`](https://pymupdf.readthedocs.io/) **or** Poppler utilities (`pdftoppm`/`pdftotext`) on your `PATH` for PDF preprocessing.
2. Activate your virtual environment if needed (not created automatically).
3. Execute the CLI with an experiment definition:

```bash
python main.py --experiment experiments/prada_2023.json
```

Key switches:
- `--model-backend {dummy,openai,google,groq,cerebras}` selects the model adapter (dummy returns a fixed label; OpenAI-compatible posts to a local/remote API; Google adapter sends requests to Gemini directly; Groq adapter calls Groq's OpenAI-compatible endpoint; Cerebras adapter targets Cerebras Inference).
- `--openai-model`, `--api-base`, `--api-key/--api-key-env`, `--temperature`, `--top-p`, `--max-tokens`, `--request-timeout`, and `--extra-header` configure the OpenAI-compatible adapter.
- `--groq-model`, `--groq-api-base`, `--groq-api-key/--groq-api-key-env`, `--temperature`, `--top-p`, `--max-tokens`, `--request-timeout`, `--extra-header`, `--retry-attempts`, and `--retry-wait` configure the Groq adapter.
- `--cerebras-model`, `--cerebras-api-base`, `--cerebras-api-key/--cerebras-api-key-env`, `--temperature`, `--top-p`, `--max-tokens`, `--request-timeout`, `--extra-header`, `--retry-attempts`, and `--retry-wait` configure the Cerebras adapter.
- `--google-model`, `--google-api-base`, `--google-api-key/--google-api-key-env`, `--temperature`, `--top-p`, `--google-top-k`, `--google-max-output-tokens`, `--google-safety`, and `--request-timeout` configure the Google adapter.
- `--retry-attempts` and `--retry-wait` control how the CLI backs off and retries when the upstream API reports rate limits (HTTP 429).
- `--benchmarks PATH`, `--artifacts DIR`, and `--experiment-id NAME` adjust inputs and artifact locations.
- `--no-images` / `--no-text` skip respective preprocessing stages when the downstream model does not need them.
- `--resume` skips tasks that already have saved responses under the target artifacts directory, letting you continue an interrupted run.

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

Update `configs/litellm_config.yaml` to match the proxied model name or sampling parameters when needed.
