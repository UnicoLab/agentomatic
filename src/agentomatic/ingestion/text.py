"""Document text helpers: normalise → section → chunk → quality.

These primitives are used by builtin ingestors and by project code that
needs structured chunks without standing up a full :class:`BaseIngestor`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class Section:
    """A markdown section delimited by a heading."""

    heading: str
    level: int
    body: str


@dataclass
class Chunk:
    """A sliding-window text chunk."""

    chunk_id: str
    index: int
    text: str


@dataclass
class IngestTextResult:
    """Result of running the text ingestion helpers on one document."""

    title: str
    file_hash: str
    markdown: str
    normalized_markdown: str
    sections: list[Section] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    quality_score: float = 0.0
    word_count: int = 0
    warnings: list[str] = field(default_factory=list)


def content_hash(data: bytes) -> str:
    """Return the first 16 hex chars of the SHA-256 of *data*."""
    return hashlib.sha256(data).hexdigest()[:16]


def convert_to_markdown(source: str, *, is_path: bool) -> str:
    """Convert a document to markdown via MarkItDown when available.

    Args:
        source: A file path (when *is_path*) or inline text/markdown.
        is_path: Whether *source* is a filesystem path.

    Returns:
        Markdown text (falls back to the raw text if conversion is unavailable).
    """
    if not is_path:
        return source
    path = Path(source)
    if not path.exists():
        return source
    try:
        from markitdown import MarkItDown

        result = MarkItDown().convert(str(path))
        return getattr(result, "text_content", "") or getattr(result, "markdown", "") or ""
    except Exception:  # noqa: BLE001 - markitdown missing/unsupported -> read text
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""


def normalize_markdown(markdown: str) -> str:
    """Clean/normalise markdown for downstream processing.

    Collapses excess blank lines and trailing whitespace, and normalises
    Windows newlines.
    """
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_sections(markdown: str) -> list[Section]:
    """Split markdown into heading-delimited sections (preamble = level 0)."""
    matches = list(_HEADING_RE.finditer(markdown))
    sections: list[Section] = []
    if not matches:
        if markdown.strip():
            sections.append(Section(heading="", level=0, body=markdown.strip()))
        return sections
    if matches[0].start() > 0:
        preamble = markdown[: matches[0].start()].strip()
        if preamble:
            sections.append(Section(heading="", level=0, body=preamble))
    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        sections.append(Section(heading=heading, level=level, body=body))
    return sections


def chunk_text(markdown: str, *, max_words: int, overlap_words: int) -> list[Chunk]:
    """Split text into overlapping word-window chunks."""
    words = markdown.split()
    if not words:
        return []
    step = max(1, max_words - overlap_words)
    chunks: list[Chunk] = []
    for idx, start in enumerate(range(0, len(words), step)):
        window = words[start : start + max_words]
        if not window:
            break
        chunks.append(Chunk(chunk_id=f"chunk-{idx}", index=idx, text=" ".join(window)))
        if start + max_words >= len(words):
            break
    return chunks


def quality_score(markdown: str, sections: list[Section]) -> float:
    """Compute an additive 0..1 quality heuristic."""
    words = len(markdown.split())
    headings = [s for s in sections if s.heading]
    bullet_lines = sum(1 for line in markdown.splitlines() if line.strip().startswith(("-", "*")))
    score = 0.0
    if words > 50:
        score += 0.3
    if words > 200:
        score += 0.1
    if sections:
        score += 0.2
    if len(sections) >= 3:
        score += 0.1
    if headings:
        score += 0.1
    if bullet_lines >= 3:
        score += 0.1
    keywords = ("module", "requirement", "scope", "feature", "api", "database")
    if any(kw in markdown.lower() for kw in keywords):
        score += 0.1
    return min(score, 1.0)


def ingest_text(
    source: str,
    *,
    is_path: bool = False,
    title: str | None = None,
    chunk_size_tokens: int | None = None,
    chunk_overlap_tokens: int | None = None,
    min_quality: float | None = None,
) -> IngestTextResult:
    """Run the full text pipeline on one document.

    Args:
        source: File path (when *is_path*) or inline text/markdown.
        is_path: Whether *source* is a filesystem path.
        title: Optional title override (defaults to the file/document name).
        chunk_size_tokens: Override platform default chunk size.
        chunk_overlap_tokens: Override platform default overlap.
        min_quality: Override platform default quality threshold.

    Returns:
        A populated :class:`IngestTextResult`.
    """
    try:
        from agentomatic.config.settings import get_settings

        cfg: Any = get_settings()
    except Exception:  # noqa: BLE001

        class _Defaults:
            chunk_size_tokens = 1200
            chunk_overlap_tokens = 150
            min_quality_score = 0.70

        cfg = _Defaults()

    size = chunk_size_tokens if chunk_size_tokens is not None else cfg.chunk_size_tokens
    overlap = (
        chunk_overlap_tokens if chunk_overlap_tokens is not None else cfg.chunk_overlap_tokens
    )
    quality_min = min_quality if min_quality is not None else cfg.min_quality_score

    raw = convert_to_markdown(source, is_path=is_path)
    normalized = normalize_markdown(raw)
    sections = extract_sections(normalized)
    max_words = max(50, int(size * 0.75))
    overlap_words = max(0, int(overlap * 0.75))
    chunks = chunk_text(normalized, max_words=max_words, overlap_words=overlap_words)
    score = quality_score(normalized, sections)
    file_hash = content_hash(normalized.encode("utf-8"))
    resolved_title = title or (Path(source).stem if is_path else "document")

    warnings: list[str] = []
    if score < quality_min:
        warnings.append(f"Low quality score ({score:.2f} < {quality_min}).")
    if not normalized.strip():
        warnings.append("Empty document after conversion.")

    return IngestTextResult(
        title=resolved_title,
        file_hash=file_hash,
        markdown=raw,
        normalized_markdown=normalized,
        sections=sections,
        chunks=chunks,
        quality_score=round(score, 4),
        word_count=len(normalized.split()),
        warnings=warnings,
    )
