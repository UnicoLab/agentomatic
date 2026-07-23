"""Shared domain library for the Scooper ai_platform.

Keeps only what Agentomatic does not provide: estimator math, artifact
versioning, embeddings / vector adapters, schemas, language helpers, audit
logging, telemetry, task-progress bridging, and JSON repair for small LLMs.

LLM clients and prompt managers come from Agentomatic (stack + per-agent
``llm.py`` + ``prompts.json`` injection into ``BaseGraphAgent``). Nodes call
:func:`agentomatic.providers.invoke_with_retry` (and optionally
``get_structured_llm``) on the injected ``self.llm`` — no parallel LLM layer.

See ``ai_core/README.md`` for the module map and when to use each helper.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
