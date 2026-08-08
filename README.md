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

config = IngestionConfig(
    azure_openai_endpoint="https://<resource>.openai.azure.com/",
    azure_openai_api_key="…",          # or set RAG_INGEST_AZURE_OPENAI_API_KEY
)
pipeline = IngestionPipeline(config)

result = pipeline.ingest_file("report.pdf")
print(result.status, result.stats.parent_strategy, len(result.chunks))

for chunk in result.chunks:
    if chunk.role.value == "child":
        doc = chunk.to_langchain()     # page_content + flat metadata, ready to embed

# whole directory, concurrently, with per-file error isolation
results = pipeline.ingest_directory("./docs", glob="**/*")
```

Embeddings are only used by the semantic chunker (pdf/docx). To run without Azure
(tabular formats, or tests), pass a mock embeddings object:
`IngestionPipeline(config, embeddings=my_fake_embeddings)`.

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

---

## Config reference

`pydantic-settings`, env prefix `RAG_INGEST_` (e.g. `RAG_INGEST_MAX_CHILD_TOKENS=600`).
A `.env` file is read if present.

| Setting | Default | Purpose |
|---|---|---|
| `azure_openai_endpoint` | `None` | Azure OpenAI endpoint (semantic chunker only) |
| `azure_openai_api_key` | `None` | Azure OpenAI key (`SecretStr`) |
| `embedding_deployment` | `text-embedding-3-large` | Embedding deployment name |
| `embedding_batch_size` | `64` | Embedding batch size |
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
| `toc_inference_deployment` | `gpt-4o-mini` | Deployment for Tier 3 |
| `max_heading_candidates` | `500` | Above this, skip Tier 3 → PAGE |
| `max_heading_chars` | `120` | Max length of a heading candidate line |
| `min_chars_per_page` | `50` | Quality gate; below this → `LowExtractionQualityError` |
| `max_workers` | `4` | `ingest_directory` thread pool size |
| `enable_embedding_cache` | `True` | Cache embeddings by content hash |
| `cache_dir` | `.ingest_cache` | Embedding cache location |

No thresholds are hardcoded in the chunkers — they all read from this object.

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

## Out of scope

Embedding, vector-store writes, index creation, retrieval/reranking, the
parent-lookup docstore, the Azure Document Intelligence fallback parser,
image/figure extraction, and delta ingestion beyond deterministic IDs.
