"""Canonical MarkItDown-supported upload formats.

Keep frontend ``accept`` attributes and the ingestor MIME→suffix map in sync
with what MarkItDown (and its optional extras) can convert.
"""

from __future__ import annotations

# Extensions MarkItDown can convert when extras are installed
# (``markitdown[pdf,docx,pptx,xlsx,xls,outlook]``).
MARKITDOWN_EXTENSIONS: tuple[str, ...] = (
    # Documents
    ".pdf",
    ".doc",
    ".docx",
    ".odt",
    ".rtf",
    ".epub",
    # Presentations
    ".ppt",
    ".pptx",
    # Spreadsheets
    ".xls",
    ".xlsx",
    ".csv",
    # Text / markup
    ".md",
    ".markdown",
    ".txt",
    ".text",
    ".html",
    ".htm",
    ".xml",
    ".json",
    ".jsonl",
    ".log",
    ".rst",
    # Notebooks / archives / mail
    ".ipynb",
    ".zip",
    ".msg",
    # Images (OCR / caption when available)
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    # Audio (transcription extras)
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
)

# HTML ``<input accept>`` value covering the list above + MIME wildcards.
MARKITDOWN_ACCEPT_ATTR: str = (
    ",".join(MARKITDOWN_EXTENSIONS)
    + ",text/*,image/*,application/pdf,"
    + "application/msword,"
    + "application/vnd.openxmlformats-officedocument.wordprocessingml.document,"
    + "application/vnd.ms-powerpoint,"
    + "application/vnd.openxmlformats-officedocument.presentationml.presentation,"
    + "application/vnd.ms-excel,"
    + "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
    + "application/json,application/xml,application/epub+zip,application/zip"
)

# MIME → suffix for temp files when the original filename has no extension.
MARKITDOWN_MIME_SUFFIX: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/epub+zip": ".epub",
    "application/zip": ".zip",
    "application/json": ".json",
    "application/xml": ".xml",
    "application/vnd.ms-outlook": ".msg",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/csv": ".csv",
    "text/html": ".html",
    "text/xml": ".xml",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
}


def is_markitdown_extension(filename: str) -> bool:
    """Return True when *filename* ends with a known MarkItDown extension."""
    lower = filename.lower().strip()
    return any(lower.endswith(ext) for ext in MARKITDOWN_EXTENSIONS)
