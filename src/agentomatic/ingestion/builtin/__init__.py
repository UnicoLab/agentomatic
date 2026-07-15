"""Built-in ingestors shipped with Agentomatic.

These ingestors provide sensible, dependency-light defaults for the most
common ops flows.  They are opt-in — nothing is registered unless the user
imports the concrete class and adds it to their registry (or uses the CLI
templates that scaffold them).
"""

from __future__ import annotations

from .markdown import MarkdownIngestor, MarkdownIngestRequest

__all__ = ["MarkdownIngestor", "MarkdownIngestRequest"]
