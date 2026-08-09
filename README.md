# rag_ingestion

Parser-agnostic, multi-format document ingestion & chunking.

**One job:** file in → `list[Chunk]` out. It does **not** embed, index, or
retrieve — those are downstream concerns and deliberately kept out of this module.

All format-specific logic terminates at the parser layer. Everything downstream
operates on a single Intermediate Representation (IR), so adding a new file format
requires **zero changes** to the chunkers or the pipeline.

```
file ──▶ Parser ──▶ IRDocument ──▶ Chunker ──▶ list[Chunk] ──▶ (downstream: embed/index)
        (format          (single           (dispatch by
         specific)        contract)         file_type)
```

Only `role == CHILD` chunks are meant to be embedded. `PARENT` chunks are stored
in a docstore (downstream) and returned at retrieval time.

---

## Quickstart

Requires Python 3.12. Uses [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

```python
from rag_ingestion import IngestionConfig, IngestionPipeline

# Local by default: sentence-transformers, no credentials needed.
config = IngestionConfig()

# ...or use Azure embeddings instead:
# config = IngestionConfig(
#     embedding_provider="azure",
#     azure_openai_endpoint="https://<resource>.openai.azure.com/",
#     azure_openai_api_key="…",        # or set RAG_INGEST_AZURE_OPENAI_API_KEY
# )
pipeline = IngestionPipeline(config)

result = pipeline.ingest_file("report.pdf")
print(result.status, result.stats.parent_strategy, len(result.chunks))

for chunk in result.chunks:
    if chunk.role.value == "child":
        doc = chunk.to_langchain()     # page_content + flat metadata, ready to embed

# whole directory, concurrently, with per-file error isolation
results = pipeline.ingest_directory("./docs", glob="**/*")
```

Embeddings are only used by the semantic chunker (pdf/docx). By default they run
**fully local** via sentence-transformers — the model downloads once from the HF
hub and is cached, and nothing leaves your machine. Switch to Azure with
`embedding_provider="azure"`, or inject any embeddings object (e.g. a mock in
tests): `IngestionPipeline(config, embeddings=my_embeddings)`.

Run the demo (offline, mocked embeddings):

```bash
python examples/ingest_demo.py
```

---

## Format → strategy

| Format | Parser | Chunker | Parent unit | Child unit |
|---|---|---|---|---|
| PDF | Docling | `SemanticParentChildChunker` | Section via the **parent-strategy ladder** (heading / page / LLM) | Semantic split within the parent |
| DOCX | Docling | `SemanticParentChildChunker` | Heading-anchored section (or one flat section) | Semantic split |
| PPTX | Docling | `SlideChunker` | ±`slide_window` slide window (or section-divider group) | One chunk per slide (title + body + notes) |
| XLSX | pandas + openpyxl | `RowChunker` | Sheet summary (columns, dtypes, sample) | Batched rows, headers preserved |
| CSV | pandas | `RowChunker` | Sheet summary | Batched rows |
| Tables (any) | — | `TableChunker` (shared) | Enclosing section's parent | One TABLE chunk (caption + markdown), header repeated on row splits |

### PDF parent-strategy ladder (§5.1.1)

Resolved per document; the winner is recorded in
`IRDocument.raw_metadata["parent_strategy"]` and surfaced in
`IngestionStats.parent_strategy`.

| Tier | Strategy | Trigger | Parent unit |
|---|---|---|---|
| 1 | `HEADING` | ≥ `min_headings_for_structure` headings **and** heading coverage ≥ `min_heading_page_coverage` of pages | Heading-anchored section |
| 2 | `PAGE` | Tier 1 fails (and LLM inference off/failed) | One section per page, with paragraph bleed |
| 3 | `LLM_INFERRED` | Tier 1 fails and `enable_llm_toc_inference` is on | Section from LLM-confirmed heading candidates |

Coverage guards the half-structured case (e.g. 2 headings across 40 pages passes a
naive count but yields pathological 20-page parents). The LLM tier classifies
heuristic heading *candidates* (it never generates a TOC), validates returned
`line_id`s against what was submitted, clamps levels 1–4, retries once on bad
JSON, and falls back to `PAGE` — it never raises.

The Tier-3 chat model defaults to **local Ollama** (`llm_provider="ollama"`,
`ollama_model="mistral"`) — run `ollama pull mistral` with the daemon up. Set
`llm_provider="azure"` to use AzureChatOpenAI instead. Tier 3 is off by default
(`enable_llm_toc_inference=False`).

---

## Config reference

`pydantic-settings`, env prefix `RAG_INGEST_` (e.g. `RAG_INGEST_MAX_CHILD_TOKENS=600`).
A `.env` file is read if present.

| Setting | Default | Purpose |
|---|---|---|
| `embedding_provider` | `sentence_transformers` | `sentence_transformers` (local) or `azure` |
| `local_embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | HF model id for local embeddings |
| `local_embedding_device` | `cpu` | `cpu` or `cuda` for local embeddings |
| `embedding_batch_size` | `64` | Embedding batch size |
| `azure_openai_endpoint` | `None` | Azure OpenAI endpoint (provider=azure only) |
| `azure_openai_api_key` | `None` | Azure OpenAI key (`SecretStr`) |
| `embedding_deployment` | `text-embedding-3-large` | Azure embedding deployment name |
| `breakpoint_threshold_type` | `percentile` | SemanticChunker breakpoint type |
| `breakpoint_threshold_amount` | `95.0` | SemanticChunker breakpoint amount |
| `max_parent_tokens` | `3000` | Split parents larger than this at paragraph boundaries |
| `max_child_tokens` | `800` | Re-split children larger than this |
| `min_child_tokens` | `100` | Merge children smaller than this into the previous sibling |
| `row_batch_target_tokens` | `400` | Target token size per row batch |
| `max_rows_per_chunk` | `10` | Hard cap on rows per row chunk |
| `max_rows_for_row_chunking` | `5000` | Above this a sheet is routed to structured-query (parent only) |
| `slide_window` | `1` | ± slides in a PPTX parent window |
| `min_headings_for_structure` | `2` | Tier 1 heading-count threshold |
| `min_heading_page_coverage` | `0.6` | Tier 1 page-coverage threshold |
| `enable_llm_toc_inference` | `False` | Enable Tier 3 LLM heading inference |
| `llm_provider` | `ollama` | Tier 3 chat LLM: `ollama` (local) or `azure` |
| `ollama_model` | `mistral` | Ollama model for Tier 3 (e.g. Mistral 7B) |
| `ollama_base_url` | `http://localhost:11434` | Ollama daemon URL |
| `toc_inference_deployment` | `gpt-4o-mini` | Azure deployment for Tier 3 (provider=azure) |
| `max_heading_candidates` | `500` | Above this, skip Tier 3 → PAGE |
| `max_heading_chars` | `120` | Max length of a heading candidate line |
| `token_counter` | `auto` | `auto` (HF when offline/local, tiktoken when online), `tiktoken`, or `hf` |
| `token_encoder_model` | `None` | HF tokenizer for the `hf` backend; defaults to `local_embedding_model` |
| `offline` | `False` | Air-gap: force offline env + no telemetry; rejects Azure providers |
| `min_chars_per_page` | `50` | Quality gate; below this → `LowExtractionQualityError` |
| `max_workers` | `4` | `ingest_directory` thread pool size |
| `enable_embedding_cache` | `True` | Cache embeddings by content hash |
| `cache_dir` | `.ingest_cache` | Embedding cache location |

No thresholds are hardcoded in the chunkers — they all read from this object.

**Token counting.** Counts drive chunk *sizing* (child/parent/table/row budgets),
so the `auto` default matches the counter to the pipeline: the local embedding
model's HF tokenizer when offline/local, `tiktoken` (`cl100k_base`) when using
Azure. The HF default is the *embedder's* tokenizer, which is the right proxy for
how children are embedded and indexed. It is **not** Mistral's tokenizer — the
Tier-3 LLM only sees short heading lines, so that doesn't matter there; but if you
size parents against a downstream local LLM and want exact counts, set
`token_encoder_model` to that model's tokenizer (e.g. a Mistral tokenizer id).

---

## How to add a new format

The parser is the only seam. To add, say, HTML:

1. **Write a parser** implementing the `DocumentParser` protocol
   (`parsers/base.py`): a `supported_extensions: set[str]` and
   `parse(path) -> IRDocument`. Map your format onto `Section` / `TextBlock` /
   `TableBlock`. Emit tables into `Section.tables`, never into `text_blocks`.
2. **Register it** in `parsers/registry.py` (or call `registry.register(...)`).
3. **Pick a chunker.** If your IR looks like prose sections, reuse
   `SemanticParentChildChunker` — just map your `file_type` to it in
   `IngestionPipeline._chunkers`. Only write a new chunker for a genuinely new
   shape (like slides or rows). Chunkers consume the IR uniformly and never branch
   on how a section was produced.

That's it — the IR contract means nothing else changes.

The `DocumentParser` protocol is also the reserved seam for an Azure Document
Intelligence fallback parser (for docs that trip the quality gate); it is
intentionally not implemented here.

---

## Testing

```bash
uv run pytest                       # fast; embeddings mocked, no Docling models
RAG_INGEST_RUN_DOCLING=1 uv run pytest tests/test_docling_integration.py
uv run mypy --strict rag_ingestion
```

Most assertions run against synthetic IR element lists, so the tier ladder,
chunkers, and IR contract are verified without downloading Docling models or
calling Azure. See `tests/` for the §9 coverage matrix.

## Air-gapped / offline operation

In the default configuration **no document content leaves the machine**:
embeddings run locally (sentence-transformers), the Tier-3 LLM talks to a local
Ollama daemon, and Docling parses in-process. The Azure providers are the only
paths that transmit your data, and they are opt-in.

The only outbound connections in the default setup are **one-time downloads of
public model/vocab files** (sentence-transformers weights, Docling OCR/TableFormer
models, and the `tiktoken` BPE vocab) — these contain none of your data. To run
fully air-gapped, pre-stage them once on a networked machine, then enforce offline
mode:

```bash
# 1) pre-cache models while online (caches under ~/.cache/huggingface, etc.)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
docling-tools models download          # Docling OCR/TableFormer models
ollama pull mistral                    # only if you enable Tier-3 TOC inference

# 2) run offline — missing assets now fail loudly instead of being fetched
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

```python
config = IngestionConfig(offline=True)   # forces offline env + no telemetry
pipeline = IngestionPipeline(config)     # raises if any provider is 'azure'
```

`offline=True` sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`, disables HF/library
telemetry and LangChain tracing, and **rejects config that pairs offline with an
Azure provider** (which would egress data). For the strongest guarantee, also
export those env vars before launching Python and/or block egress at the network
layer.

## Microservice + Streamlit UI

An optional local service wraps the module with a job queue and a small UI, for
ingesting files without writing Python. It adds orchestration only — parsing and
chunking still come entirely from `rag_ingestion`, and everything stays on the
machine (workers run with `offline=True`).

```
Streamlit ──▶ FastAPI ──▶ Celery task ──▶ Redis ──▶ worker(s) ──▶ output/<job>/<file>.jsonl
   (UI)      POST /jobs                    (queue)   (pipeline)          ▲
                │                                                        │
                └──────────── SQLite (jobs + job_files, SQLAlchemy) ◀────┘
                                     GET /jobs/{id}  (percent progress)
```

- **Batch jobs.** One upload of N files = one `job` with N `job_files`; each file
  is its own Celery task, so one bad file never aborts the batch.
- **Graceful under load.** `acks_late` + `prefetch_multiplier=1`: a worker holds
  one heavy file at a time and re-queues it on crash. Scale by adding workers.
- **Progress.** Percent = mean of per-file progress (queued 0 / running 50 /
  done 100), polled via `GET /jobs/{id}`.
- **Storage.** Chunks are written to `output/<job>/<file>.jsonl`; SQLite keeps
  only job/file progress + stats (SQLAlchemy ORM, Alembic migrations).

### API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/jobs` | multipart upload → `{id, status, total_files}` (202) |
| `GET` | `/jobs` | recent job summaries |
| `GET` | `/jobs/{id}` | job + per-file status and percent |
| `GET` | `/jobs/{id}/chunks` | produced chunks (`?file_id`, `?limit`, `?offset`) |
| `GET` | `/health` | liveness |

### Run with Docker (recommended)

Fully local and offline at runtime. Models come from a prefetched named volume;
`api`/`worker` run with `HF_HUB_OFFLINE=1` and `RAG_INGEST_OFFLINE=true`.

```bash
make prefetch     # ONCE, online: downloads models into the model-cache volume
make up           # build + start redis, api, worker, ui (offline)
# UI:  http://localhost:8501      API docs: http://localhost:8000/docs
make logs         # tail;   make down to stop
```

The only network access is `make prefetch` (public model/vocab files, no
document data). After that, unplug the network and it still runs.

### Run locally (bare processes)

Needs a local Redis and the `service` extras (`uv pip install -e ".[service,dev]"`):

```bash
make migrate       # alembic upgrade head  (creates the SQLite schema)
make dev-api       # uvicorn service.api:app --reload   (:8000)
make dev-worker    # celery -A service.celery_app worker (another terminal)
make dev-ui        # streamlit run ui/app.py             (:8501)
```

### Migrations

The DB is defined with SQLAlchemy 2.0 models (`service/models.py`) and versioned
with Alembic (`alembic/`). Change a model, then:

```bash
make revision m="add foo column"   # autogenerate
make migrate                       # apply
```

Swapping SQLite for Postgres is only a `RAG_SVC_DATABASE_URL` change — the ORM
and migrations are unchanged.

## Out of scope

Embedding, vector-store writes, index creation, retrieval/reranking, the
parent-lookup docstore, the Azure Document Intelligence fallback parser,
image/figure extraction, and delta ingestion beyond deterministic IDs.
