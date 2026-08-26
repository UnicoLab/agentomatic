"""Record invocation history for agents, plugins, pipelines, and more."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger

from agentomatic.logs.runtime import get_pipeline_log_name, normalize_resource_type

if TYPE_CHECKING:
    from agentomatic.storage.base import BaseStore

# Soft caps to keep row sizes reasonable for DB + LLM analysis budgets.
_DEFAULT_MAX_JSON_CHARS = 32_000
_DEFAULT_MAX_STRING_CHARS = 8_000
_DEFAULT_MAX_DEPTH = 6
_DEFAULT_MAX_LIST_ITEMS = 50
_AUDIT_REFERENCE_KEY = (
    os.getenv("AGENTOMATIC_AUDIT_HASH_KEY")
    or os.getenv("AGENTOMATIC_API_KEY")
    or secrets.token_hex(32)
).encode("utf-8")


def _audit_thread_reference(thread_id: str | None) -> str | None:
    """Return a stable, non-reversible reference suitable for an audit sink.

    Conversation identifiers can be user-provided and are therefore not safe
    audit metadata.  Retain a short correlation reference without copying the
    source value to a file or external SIEM.
    """
    if not thread_id:
        return None
    return hmac.new(_AUDIT_REFERENCE_KEY, thread_id.encode("utf-8"), hashlib.sha256).hexdigest()[
        :16
    ]


def truncate_for_storage(
    value: Any,
    *,
    max_chars: int = _DEFAULT_MAX_JSON_CHARS,
    max_string: int = _DEFAULT_MAX_STRING_CHARS,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    max_list_items: int = _DEFAULT_MAX_LIST_ITEMS,
    _depth: int = 0,
) -> Any:
    """Return a JSON-serialisable, size-bounded copy of ``value``.

    Large strings/lists/dicts are truncated with a marker so invocation
    logs remain useful without blowing up storage or analyser prompts.

    Args:
        value: Arbitrary input/output/metadata payload.
        max_chars: Approximate character budget for nested structures.
        max_string: Max characters retained per string leaf.
        max_depth: Max nesting depth before replacing with a marker.
        max_list_items: Max items retained per list/tuple.
        _depth: Internal recursion depth.

    Returns:
        A truncated, JSON-friendly structure.
    """
    if _depth >= max_depth:
        return "<truncated:max_depth>"

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        if len(value) <= max_string:
            return value
        return value[:max_string] + f"...<truncated:{len(value) - max_string} chars>"

    if isinstance(value, (bytes, bytearray)):
        return f"<bytes:{len(value)}>"

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        used = 2
        for key, item in value.items():
            key_s = str(key)
            truncated = truncate_for_storage(
                item,
                max_chars=max_chars,
                max_string=max_string,
                max_depth=max_depth,
                max_list_items=max_list_items,
                _depth=_depth + 1,
            )
            # Rough size accounting — stop early if budget exhausted.
            approx = len(key_s) + 4 + len(str(truncated))
            if used + approx > max_chars and out:
                out["__truncated__"] = f"{len(value) - len(out)} keys omitted"
                break
            out[key_s] = truncated
            used += approx
        return out

    if isinstance(value, (list, tuple)):
        items = list(value)
        kept = [
            truncate_for_storage(
                item,
                max_chars=max_chars,
                max_string=max_string,
                max_depth=max_depth,
                max_list_items=max_list_items,
                _depth=_depth + 1,
            )
            for item in items[:max_list_items]
        ]
        if len(items) > max_list_items:
            kept.append(f"<truncated:{len(items) - max_list_items} items>")
        return kept

    # Fallback for pydantic models / custom objects
    if hasattr(value, "model_dump"):
        try:
            return truncate_for_storage(
                value.model_dump(),
                max_chars=max_chars,
                max_string=max_string,
                max_depth=max_depth,
                max_list_items=max_list_items,
                _depth=_depth + 1,
            )
        except Exception:  # noqa: BLE001
            pass

    text = str(value)
    if len(text) > max_string:
        return text[:max_string] + f"...<truncated:{len(text) - max_string} chars>"
    return text


def _safe_dump(obj: Any) -> dict[str, Any]:
    """Best-effort convert request/response objects to a dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            dumped = obj.model_dump()
            return dumped if isinstance(dumped, dict) else {"value": dumped}
        except Exception:  # noqa: BLE001
            return {"value": str(obj)}
    return {"value": str(obj)}


class InvocationLogRecorder:
    """Write invocation logs when ``logs_history`` is enabled.

    Nil-checks the store and never raises into the request path — logging
    failures are swallowed after a warning so request behaviour is unchanged.
    """

    def __init__(self, store: BaseStore | None) -> None:
        """Attach an optional :class:`~agentomatic.storage.base.BaseStore`.

        Args:
            store: Persistence backend. When ``None``, all writes are no-ops.
        """
        self._store = store

    @property
    def enabled(self) -> bool:
        """Return ``True`` when a store is available for writes."""
        return self._store is not None

    async def record(
        self,
        *,
        resource_type: str = "agent",
        resource_name: str | None = None,
        agent_name: str | None = None,
        endpoint: str,
        input_data: Any = None,
        output_data: Any = None,
        metadata: dict[str, Any] | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
        status: str = "ok",
    ) -> dict[str, Any] | None:
        """Persist one invocation log entry.

        Args:
            resource_type: ``agent`` | ``plugin`` | ``pipeline`` |
                ``ingestion`` | ``endpoint``.
            resource_name: Resource identifier (preferred).
            agent_name: Backward-compatible alias for ``resource_name``
                (used by existing agent routes).
            endpoint: Operation label (``invoke``, ``predict``, ``run``,
                ``pipeline_step``, …).
            input_data: Request payload (truncated before storage).
            output_data: Response payload (truncated before storage).
            metadata: Extra metadata (prompt version, pipeline step, etc.).
            thread_id: Optional conversation thread.
            run_id: Optional run/correlation id (auto-generated if omitted).
            error: Error message when ``status`` is not ``ok``.
            duration_ms: Wall-clock duration in milliseconds.
            status: ``ok``, ``error``, or ``suspended``.

        Returns:
            Stored log dict, or ``None`` when the store is unavailable.
        """
        if self._store is None:
            return None

        name = resource_name or agent_name
        if not name:
            logger.warning("Invocation log skipped: missing resource_name/agent_name")
            return None

        try:
            rtype = normalize_resource_type(resource_type)
        except ValueError as exc:
            logger.warning("{}", exc)
            return None

        meta = dict(metadata or {})
        pipeline_name = get_pipeline_log_name()
        if pipeline_name and "pipeline" not in meta:
            meta["pipeline"] = pipeline_name

        try:
            run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
            entry = await self._store.create_invocation_log(
                agent_name=name,
                resource_type=rtype,
                thread_id=thread_id,
                run_id=run_id,
                endpoint=endpoint,
                input_data=truncate_for_storage(_safe_dump(input_data)),
                output_data=truncate_for_storage(_safe_dump(output_data)),
                metadata=truncate_for_storage(meta),
                error=error,
                duration_ms=duration_ms,
                status=status,
            )
            # The durable invocation record is also the canonical safe audit
            # event: it contains only bounded metadata and a generated run ID,
            # never raw request bodies or credentials.  File-sink failure must
            # not affect a serving request, just like DB log failure above.
            try:
                from agentomatic.observability.audit import emit_audit_event

                emit_audit_event(
                    agent=name,
                    op=f"{rtype}:{endpoint}",
                    request_id=run_id,
                    outcome={"ok": "success", "suspended": "suspended"}.get(status, "error"),
                    latency_ms=duration_ms,
                    extra={
                        "resource_type": rtype,
                        "thread_ref": _audit_thread_reference(thread_id),
                    },
                )
            except Exception as audit_exc:  # noqa: BLE001 - audit must not break serving.
                logger.warning(
                    "Failed to emit audit event for '{}:{}': {}", rtype, name, audit_exc
                )
            return entry
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to persist invocation log for '{}:{}': {}",
                rtype,
                name,
                exc,
            )
            return None
