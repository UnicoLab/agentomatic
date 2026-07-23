"""Built-in markdown ingestor.

Converts an input document (PDF, DOCX, HTML, plain text, or already-markdown)
into a markdown file on disk that downstream steps — typically a fan-out map
of extraction agents — can consume.

The heavy lifting is optional: this ingestor uses ``pymupdf4llm`` when
available (best PDF quality), then ``docling`` as a rich fallback, and
finally reads the file as UTF-8 text so it always produces *some* output.
Missing dependencies are soft-imported so the base package stays lean.

Example::

    from agentomatic.ingestion.builtin import MarkdownIngestor

    registry.register(MarkdownIngestor())

    # POST /api/v1/ingestion/markdown/run
    # {"source": "/tmp/report.pdf", "output_dir": "/tmp/md"}
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentomatic.ingestion.base import BaseIngestor
from agentomatic.ingestion.context import IngestionContext
from agentomatic.ingestion.models import IngestionResult


class MarkdownIngestRequest(BaseModel):
    """Input model for :class:`MarkdownIngestor`.

    Attributes:
        source: Filesystem path to the document to convert. Supported
            extensions include ``.pdf``, ``.docx``, ``.html``/``.htm``,
            ``.txt`` and ``.md``.  Anything else is read as UTF-8.
        output_dir: Directory where the resulting markdown file is written.
            Defaults to ``./ingested_markdown`` and is created if missing.
        output_filename: Optional explicit filename for the markdown output.
            Defaults to ``<source stem>.md``.
        collection: Optional logical collection / index name — forwarded to
            downstream steps via :attr:`IngestionResult.collection`.
        engine: Preferred conversion engine (``"auto"``, ``"pymupdf4llm"``,
            ``"docling"``, ``"markitdown"``, ``"plain"``). ``"auto"`` picks
            the best available.
    """

    source: str = Field(..., description="Path to the document to convert.")
    output_dir: str = Field(
        default="ingested_markdown",
        description="Directory in which to write the resulting markdown file.",
    )
    output_filename: str | None = Field(
        default=None,
        description="Explicit output filename (defaults to <source stem>.md).",
    )
    collection: str | None = Field(default=None)
    engine: str = Field(
        default="auto",
        description="Conversion engine: auto | pymupdf4llm | docling | markitdown | plain.",
    )


class MarkdownIngestor(BaseIngestor[MarkdownIngestRequest]):
    """Convert a document to markdown and persist it to disk.

    The concrete conversion strategy is chosen dynamically so this ingestor
    works even when no optional dependencies are installed:

    1. When ``engine == "pymupdf4llm"`` (or ``"auto"`` for ``.pdf`` inputs)
       and ``pymupdf4llm`` is importable, use it — the highest-quality PDF
       result.
    2. When ``docling`` is importable, use its ``DocumentConverter`` — a
       broad-spectrum parser that also handles DOCX/HTML/etc.
    3. Otherwise, read the file as UTF-8 and treat its content as markdown.
    """

    ingestor_name = "markdown"
    ingestor_description = (
        "Convert PDF/DOCX/HTML/txt documents to markdown and write them to disk "
        "for downstream extraction agents."
    )
    ingestor_version = "1.0.0"

    async def ingest(
        self,
        request: MarkdownIngestRequest,
        ctx: IngestionContext,
    ) -> IngestionResult:
        """Convert ``request.source`` to markdown and write it to disk.

        Args:
            request: Parsed :class:`MarkdownIngestRequest`.
            ctx: Progress/cancellation handle.

        Returns:
            An :class:`IngestionResult` whose ``output`` contains
            ``{"path": ..., "engine": ..., "size_bytes": ...}``.
        """
        t0 = time.perf_counter()
        source_path = Path(request.source).expanduser()
        if not source_path.exists():
            return IngestionResult(
                ingestor=self.ingestor_name,
                status="failed",
                errors=[f"Source file not found: {source_path}"],
            )

        await ctx.report(
            message=f"Reading {source_path.name}",
            current=0,
            total=3,
            stage="read",
        )

        engine, markdown = self._convert_to_markdown(source_path, request.engine)

        if ctx.cancelled:
            return IngestionResult(
                ingestor=self.ingestor_name,
                status="cancelled",
                errors=["Cancelled before writing markdown"],
            )

        await ctx.report(
            message="Writing markdown output",
            current=2,
            total=3,
            stage="write",
        )

        out_dir = Path(request.output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = request.output_filename or f"{source_path.stem}.md"
        out_path = out_dir / out_name
        out_path.write_text(markdown, encoding="utf-8")

        duration = (time.perf_counter() - t0) * 1000
        await ctx.report(
            message=f"Wrote {out_path}",
            current=3,
            total=3,
            stage="done",
            path=str(out_path),
        )
        return IngestionResult(
            ingestor=self.ingestor_name,
            status="succeeded",
            documents=1,
            chunks=1,
            upserted=0,
            collection=request.collection,
            duration_ms=duration,
            output={
                "path": str(out_path),
                "engine": engine,
                "size_bytes": len(markdown.encode("utf-8")),
                "source": str(source_path),
            },
            stats={"markdown_chars": len(markdown)},
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _convert_to_markdown(self, source: Path, engine: str) -> tuple[str, str]:
        """Return ``(engine_used, markdown_text)`` for ``source``.

        Falls back to plain UTF-8 read when no rich converter is available.
        """
        chosen = engine.lower().strip()
        suffix = source.suffix.lower()

        if chosen in ("auto", "pymupdf4llm") and suffix == ".pdf":
            md = self._try_pymupdf4llm(source)
            if md is not None:
                return "pymupdf4llm", md

        if chosen in ("auto", "docling"):
            md = self._try_docling(source)
            if md is not None:
                return "docling", md

        if chosen in ("auto", "markitdown"):
            md = self._try_markitdown(source)
            if md is not None:
                return "markitdown", md

        if chosen == "pymupdf4llm":
            md = self._try_pymupdf4llm(source)
            if md is not None:
                return "pymupdf4llm", md

        return "plain", self._read_plain(source)

    @staticmethod
    def _try_markitdown(source: Path) -> str | None:
        """Try to convert with ``markitdown`` (returns ``None`` on failure)."""
        try:
            from markitdown import MarkItDown

            result = MarkItDown().convert(str(source))
            text = getattr(result, "text_content", None) or getattr(result, "markdown", None)
            return str(text) if text else None
        except Exception:  # noqa: BLE001 - optional dep
            return None

    @staticmethod
    def _try_pymupdf4llm(source: Path) -> str | None:
        """Try to convert with ``pymupdf4llm`` (returns ``None`` on failure)."""
        try:
            import pymupdf4llm  # type: ignore[import-not-found]
        except Exception:  # noqa: BLE001 - optional dependency
            return None
        try:
            return pymupdf4llm.to_markdown(str(source))
        except Exception:  # noqa: BLE001 - keep going with fallbacks
            return None

    @staticmethod
    def _try_docling(source: Path) -> str | None:
        """Try to convert with ``docling`` (returns ``None`` on failure)."""
        try:
            from docling.document_converter import (  # type: ignore[import-not-found]
                DocumentConverter,
            )
        except Exception:  # noqa: BLE001 - optional dependency
            return None
        try:
            converter = DocumentConverter()
            result = converter.convert(str(source))
            doc = getattr(result, "document", result)
            if hasattr(doc, "export_to_markdown"):
                return doc.export_to_markdown()
            return str(doc)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _read_plain(source: Path) -> str:
        """Fallback: read the file as UTF-8, mildly tolerant to binary data."""
        try:
            return source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return source.read_bytes().decode("utf-8", errors="replace")

    def info(self) -> dict[str, Any]:
        """Extend base info with optional-dependency status."""
        base = super().info()
        base["engines_available"] = {
            "pymupdf4llm": _module_available("pymupdf4llm"),
            "docling": _module_available("docling"),
        }
        return base


def _module_available(module_name: str) -> bool:
    """Return ``True`` when *module_name* can be imported."""
    try:
        __import__(module_name)
    except Exception:  # noqa: BLE001
        return False
    return True
