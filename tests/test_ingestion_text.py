"""Tests for ingestion text helpers and MarkItDown format constants."""

from __future__ import annotations

from agentomatic.ingestion.formats import (
    MARKITDOWN_EXTENSIONS,
    is_markitdown_extension,
)
from agentomatic.ingestion.text import (
    chunk_text,
    extract_sections,
    ingest_text,
    normalize_markdown,
    quality_score,
)


def test_normalize_markdown() -> None:
    """Excess blank lines and trailing spaces are collapsed."""
    raw = "Hello  \r\n\r\n\r\nWorld  \n"
    assert normalize_markdown(raw) == "Hello\n\nWorld"


def test_extract_sections() -> None:
    """Heading-delimited sections are extracted."""
    md = "Preamble\n\n# Title\n\nBody\n\n## Sub\n\nMore"
    sections = extract_sections(md)
    assert sections[0].level == 0
    assert sections[1].heading == "Title"
    assert sections[2].heading == "Sub"


def test_chunk_overlap() -> None:
    """Chunks overlap by the requested word count."""
    words = " ".join(f"w{i}" for i in range(20))
    chunks = chunk_text(words, max_words=10, overlap_words=2)
    assert len(chunks) >= 2
    first_tail = chunks[0].text.split()[-2:]
    second_head = chunks[1].text.split()[:2]
    assert first_tail == second_head


def test_quality_score_thresholds() -> None:
    """Short empty-ish text scores low; structured text scores higher."""
    low = quality_score("hi", [])
    assert low < 0.5
    md = (
        "# Scope\n\n"
        + " ".join(["module"] * 60)
        + "\n\n## Feature\n\n- a\n- b\n- c\n\n## API\n\nDatabase requirement."
    )
    sections = extract_sections(md)
    high = quality_score(md, sections)
    assert high >= 0.7


def test_ingest_text_pipeline() -> None:
    """ingest_text returns chunks and a hash."""
    result = ingest_text(
        "# Hello\n\n" + "word " * 80,
        chunk_size_tokens=40,
        chunk_overlap_tokens=4,
        min_quality=0.0,
    )
    assert result.word_count > 0
    assert result.chunks
    assert result.file_hash


def test_markitdown_extension() -> None:
    """Known extensions are recognised."""
    assert is_markitdown_extension("report.PDF")
    assert ".pdf" in MARKITDOWN_EXTENSIONS
    assert not is_markitdown_extension("binary.exe")
