# Ingestion & RAG

Agentomatic treats document ingestion the way it treats everything else: as an
**ops problem, not an implementation problem**. You bring the ingestion logic —
written with whatever libraries you already trust (`docling`, `unstructured`,
`pymupdf4llm`, `langchain-text-splitters`, your embedding model, your vector DB
client) — and Agentomatic packages it as a first-class, deployable resource.

!!! tip "Philosophy"
    Agentomatic provides the **packaging**: discovery, REST endpoints,
    task/queue execution with live progress and cancellation, health/status
    integration, and deployment. You still bring the parser / embedder /
    vector DB you prefer. Optional batteries — ``ingest_text`` /
    ``normalize_markdown`` / ``chunk_text`` / ``quality_score`` and the
    ``MARKITDOWN_*`` format constants — live under ``agentomatic.ingestion``
    so you do not re-write the same helpers in every project.

### Text helpers (batteries)

```python
from agentomatic.ingestion import ingest_text, MARKITDOWN_ACCEPT_ATTR

result = ingest_text(path_or_markdown, is_path=True)
# result.sections, result.chunks, result.quality_score, result.warnings
```

The builtin ``MarkdownIngestor`` also accepts ``engine="markitdown"`` (soft
import) alongside ``pymupdf4llm`` / ``docling`` / ``plain``.

## What you get

Drop a `BaseIngestor` subclass into your project and Agentomatic automatically
exposes:

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/ingestion/{name}/run` | Run synchronously, return the result |
| `POST /api/v1/ingestion/{name}/run/async` | Submit as a background **task**, return a pollable id |
| `GET /api/v1/ingestion/{name}/info` | Ingestor metadata + input schema |
| `GET /api/v1/ingestion/{name}/health` | Readiness/health |
| `GET /api/v1/ingestion` | List all ingestors |

Async runs plug straight into the [unified task system](tasks.md), so you get
progress percentages, live SSE event streams, cancellation, and webhooks for
free.

## Write an ingestor

You implement exactly one method — `ingest` — using any libraries you like:

```python
from __future__ import annotations

from pydantic import BaseModel, Field

from agentomatic.ingestion import BaseIngestor, IngestionResult


class PdfRequest(BaseModel):
    source: str = Field(..., description="Path or glob to ingest")
    collection: str = Field("default")


class PdfIngestor(BaseIngestor[PdfRequest]):
    ingestor_name = "pdf_docs"
    ingestor_description = "Parse PDFs to markdown and upsert to Qdrant."

    async def setup(self) -> None:
        from qdrant_client import AsyncQdrantClient

        from agentomatic.providers.embeddings import get_embeddings

        self.client = AsyncQdrantClient(url="http://localhost:6333")
        self.embedder = get_embeddings("openai")

    async def ingest(self, request: PdfRequest, ctx) -> IngestionResult:
        import pymupdf4llm  # your choice of parser
        from langchain_text_splitters import MarkdownTextSplitter

        markdown = pymupdf4llm.to_markdown(request.source)
        chunks = MarkdownTextSplitter().split_text(markdown)

        upserted = 0
        for i, chunk in enumerate(chunks):
            if ctx.cancelled:            # cooperative cancellation
                break
            vector = self.embedder.embed_query(chunk)
            # await self.client.upsert(request.collection, ...)
            upserted += 1
            await ctx.report(            # live progress for the frontend
                current=i + 1,
                total=len(chunks),
                message=f"upserting {i + 1}/{len(chunks)}",
            )

        return IngestionResult(
            documents=1,
            chunks=len(chunks),
            upserted=upserted,
            collection=request.collection,
        )
```

### The execution context

The `ctx` argument implements progress reporting and cancellation:

- `await ctx.report(percent=..., message=..., current=..., total=..., stage=...)`
  emits a progress event (used to compute `percent` automatically from
  `current`/`total`).
- `ctx.cancelled` returns `True` when a caller cancels the task — check it in
  long loops and exit gracefully.

When the ingestor runs synchronously, a no-op context is supplied, so the same
code works in both modes.

## Install & run

Scaffold one with the CLI:

```bash
agentomatic init pdf_docs --template ingestion
```

Move the generated folder into your project's `ingestion/` directory:

```
my-project/
├── agents/
├── plugins/
└── ingestion/
    └── pdf_docs/
        ├── __init__.py
        └── ingestor.py
```

Then start the platform — ingestors are auto-discovered:

```bash
agentomatic run
```

Run it synchronously:

```bash
curl -X POST http://localhost:8000/api/v1/ingestion/pdf_docs/run \
  -H 'content-type: application/json' \
  -d '{"source": "./docs/report.pdf", "collection": "kb"}'
```

Or as a background task with live progress:

```bash
# Submit
curl -X POST http://localhost:8000/api/v1/ingestion/pdf_docs/run/async \
  -H 'content-type: application/json' \
  -d '{"source": "./docs", "collection": "kb"}'
# -> {"id": "task_...", "status": "queued", "links": {...}}

# Poll
curl http://localhost:8000/api/v1/tasks/task_xxx

# Stream progress (SSE)
curl -N http://localhost:8000/api/v1/tasks/task_xxx/events
```

## Use it as a pipeline step

Ingestors are first-class pipeline steps, so an ingest → index → answer flow is
a single declarative pipeline. Reference an ingestor with the `ingestion:` key:

```yaml
name: rag_ingest_and_answer
steps:
  - ingestion: pdf_docs          # runs your ingestor
    name: load
    input:
      source: $.input.path
      collection: knowledge_base
  - agent: rag_agent             # downstream step sees the ingestion result
    name: answer
    input:
      query: $.input.question
```

The step stores the `IngestionResult` in the pipeline context (as
`ctx.current`), so later steps can branch on `chunks`, `upserted`, etc. Ingestion
steps support the same `input`/`output` mappings, `condition`, `on_error`,
`retry`, and `timeout` options as every other step, and the whole pipeline can
itself be run synchronously or as a background task.

## Embeddings helper

Agentomatic ships a small, cached embeddings factory (a thin adapter over
existing libraries — you still choose the backend):

```python
from agentomatic.providers.embeddings import get_embeddings

emb = get_embeddings("openai", model="text-embedding-3-small")
vectors = emb.embed_documents(["hello", "world"])
```

Providers: `ollama`, `openai`, `hash` (deterministic, dependency-free — great
for tests/offline), and `dummy`. Instances are cached per `(provider, kwargs)`,
so requesting the same configuration returns the same object.

## Programmatic registration

You can also register an ingestor without a folder:

```python
from agentomatic import AgentPlatform

platform = AgentPlatform()
platform.register_ingestor(PdfIngestor())
app = platform.build()
```

## Result model

`IngestionResult` is intentionally flexible — populate the counters that matter
and use `stats` / `output` for anything else:

| Field | Meaning |
|-------|---------|
| `documents` | Source documents processed |
| `chunks` | Chunks produced |
| `upserted` | Records written to the store |
| `skipped` | Records skipped (dedup/filters) |
| `duration_ms` | Auto-filled if you leave it at `0` |
| `collection` | Target collection/index |
| `errors` | Non-fatal error messages |
| `stats` / `output` | Free-form extra telemetry |
