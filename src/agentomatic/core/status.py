"""Unified platform status — structured endpoint + HTML dashboard.

Aggregates the health of *every* resource the platform manages (agents,
plugins, custom endpoints, ingestors, pipelines), plus the task executor and
storage backend, into a single payload. A self-contained, auto-refreshing HTML
dashboard renders that payload at ``/status``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

if TYPE_CHECKING:
    from .platform import AgentPlatform

_TAG = "Status"


def _classify(section: dict[str, dict[str, Any]], ok: tuple[str, ...]) -> dict[str, Any]:
    """Summarise a resource section into counts + a health rollup."""
    total = len(section)
    healthy = sum(1 for v in section.values() if v.get("status") in ok)
    return {
        "total": total,
        "healthy": healthy,
        "degraded": total - healthy,
        "items": section,
    }


async def build_status_payload(platform: AgentPlatform) -> dict[str, Any]:
    """Collect a complete status snapshot of the platform."""
    # Agents
    agents: dict[str, Any] = {}
    for name, agent in platform.registry.all().items():
        try:
            health = await agent.health_check()
        except Exception as exc:  # noqa: BLE001
            health = {"status": "error", "error": str(exc)}
        health.setdefault("version", getattr(agent.manifest, "version", "?"))
        agents[name] = health

    # Plugins
    plugins: dict[str, Any] = {}
    for name, plugin in platform._plugin_registry.list_plugins().items():
        plugins[name] = {
            "status": "healthy" if getattr(plugin, "is_loaded", True) else "unloaded",
            "version": getattr(plugin, "plugin_version", "?"),
            "loaded_at": getattr(plugin, "loaded_at", None),
        }

    # Custom endpoints
    endpoints: dict[str, Any] = {}
    for name, endpoint in platform._endpoint_registry.list_endpoints().items():
        try:
            endpoints[name] = await endpoint.health_check()
        except Exception as exc:  # noqa: BLE001
            endpoints[name] = {"status": "error", "error": str(exc)}

    # Ingestors
    ingestors: dict[str, Any] = {}
    for name, ingestor in platform._ingestion_registry.list_ingestors().items():
        try:
            ingestors[name] = await ingestor.health_check()
        except Exception as exc:  # noqa: BLE001
            ingestors[name] = {"status": "error", "error": str(exc)}

    # Pipelines (config-only; presence == available)
    pipelines: dict[str, Any] = {
        name: {
            "status": "available",
            "version": getattr(cfg, "version", "?"),
            "steps": len(getattr(cfg, "steps", []) or []),
        }
        for name, cfg in platform.pipelines.items()
    }

    # Storage
    storage: dict[str, Any] = {"status": "not_configured"}
    if platform.store is not None:
        try:
            storage = await platform.store.health_check()
        except Exception as exc:  # noqa: BLE001
            storage = {"status": "unhealthy", "error": str(exc)}

    # Tasks
    tasks: dict[str, Any] = {"enabled": False}
    if platform.task_manager is not None:
        try:
            tasks = {"enabled": True, **(await platform.task_manager.stats())}
        except Exception as exc:  # noqa: BLE001
            tasks = {"enabled": True, "error": str(exc)}

    # Connections (platform + per-agent scopes)
    connections: dict[str, Any] = {}
    try:
        from agentomatic.connections.manager import all_managers

        for scope, mgr in all_managers().items():
            try:
                names = mgr.list_names() if hasattr(mgr, "list_names") else list(mgr._connections)
            except Exception:  # noqa: BLE001
                names = list(getattr(mgr, "_connections", {}).keys())
            for conn_name in names:
                key = f"{scope}/{conn_name}"
                try:
                    conn = mgr.get(conn_name)
                    cfg = getattr(conn, "config", None)
                    kind = getattr(cfg, "kind", None)
                    purpose = getattr(cfg, "purpose", None)
                    connections[key] = {
                        "status": "configured",
                        "scope": scope,
                        "kind": str(getattr(kind, "value", kind) or "?"),
                        "purpose": str(getattr(purpose, "value", purpose) or "?"),
                    }
                except Exception as exc:  # noqa: BLE001
                    connections[key] = {
                        "status": "error",
                        "error": str(exc),
                        "scope": scope,
                    }
    except Exception as exc:  # noqa: BLE001
        connections["_error"] = {"status": "error", "error": str(exc)}

    sections = {
        "agents": _classify(agents, ok=("healthy",)),
        "plugins": _classify(plugins, ok=("healthy",)),
        "endpoints": _classify(endpoints, ok=("healthy", "ok")),
        "ingestors": _classify(ingestors, ok=("healthy", "not_ready")),
        "pipelines": _classify(pipelines, ok=("available",)),
        "connections": _classify(connections, ok=("configured", "healthy", "ok")),
    }

    degraded = any(s["degraded"] for s in sections.values())
    storage_ok = storage.get("status") in ("healthy", "ok", "not_configured")
    overall = "healthy" if not degraded and storage_ok else "degraded"

    return {
        "status": overall,
        "platform": {
            "name": platform.title,
            "version": platform.version,
            "uptime_seconds": round(platform._control_state.uptime_seconds, 1),
            "maintenance_mode": platform._control_state.maintenance_mode,
        },
        "summary": {
            name: {"total": s["total"], "healthy": s["healthy"]} for name, s in sections.items()
        },
        "resources": sections,
        "tasks": tasks,
        "storage": storage,
        "generated_at": time.time(),
    }


def create_status_router(platform: AgentPlatform) -> APIRouter:
    """Build the status router (JSON endpoint + HTML dashboard)."""
    router = APIRouter()

    @router.get(
        f"{platform.api_prefix}/status",
        tags=[_TAG],
        summary="Unified platform status (JSON)",
    )
    async def status_json() -> dict[str, Any]:
        """Return a structured status snapshot of the whole platform."""
        return await build_status_payload(platform)

    @router.get(
        "/status",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def status_html() -> HTMLResponse:
        """Serve the self-contained, auto-refreshing status dashboard."""
        return HTMLResponse(_STATUS_HTML.replace("__API_PREFIX__", platform.api_prefix))

    return router


# ---------------------------------------------------------------------------
# Self-contained dashboard (no external assets; polls the JSON endpoint)
# ---------------------------------------------------------------------------

_STATUS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Agentomatic — Status</title>
<style>
  :root {
    --bg: #0b0e14; --panel: #151a23; --border: #232a36; --text: #e6edf3;
    --muted: #8b98a9; --ok: #2ea043; --warn: #d29922; --err: #f85149;
    --accent: #58a6ff;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  header { padding: 24px 32px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; }
  h1 { font-size: 20px; margin: 0; display: flex; align-items: center; gap: 10px; }
  .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
  .ok { background: var(--ok); } .warn { background: var(--warn); } .err { background: var(--err); }
  .muted { color: var(--muted); font-size: 13px; }
  main { padding: 24px 32px; max-width: 1200px; margin: 0 auto; }
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 12px; margin-bottom: 28px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px; }
  .card .n { font-size: 28px; font-weight: 700; }
  .card .l { color: var(--muted); font-size: 13px; text-transform: capitalize; }
  .card .sub { font-size: 12px; margin-top: 4px; }
  section h2 { font-size: 15px; text-transform: uppercase; letter-spacing: .05em;
    color: var(--muted); margin: 24px 0 10px; }
  table { width: 100%; border-collapse: collapse; background: var(--panel);
    border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
  th, td { text-align: left; padding: 10px 14px; font-size: 14px;
    border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; }
  tr:last-child td { border-bottom: none; }
  .badge { display: inline-flex; align-items: center; gap: 6px; font-size: 12px;
    padding: 3px 9px; border-radius: 20px; background: #1f2733; }
  .badge.ok { color: var(--ok); } .badge.warn { color: var(--warn); } .badge.err { color: var(--err); }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 720px) { .grid2 { grid-template-columns: 1fr; } }
  code { background: #1f2733; padding: 2px 6px; border-radius: 5px; font-size: 12px; }
  .empty { color: var(--muted); font-size: 13px; padding: 12px 0; }
</style>
</head>
<body>
<header>
  <h1><span id="overall-dot" class="dot warn"></span> <span id="title">Agentomatic</span></h1>
  <div class="muted">
    <span id="version"></span> · uptime <span id="uptime">–</span> ·
    updated <span id="updated">–</span> · <span id="countdown"></span>
  </div>
</header>
<main>
  <div class="cards" id="cards"></div>
  <div class="grid2">
    <section><h2>Tasks</h2><div id="tasks"></div></section>
    <section><h2>Storage</h2><div id="storage"></div></section>
  </div>
  <div id="sections"></div>
</main>
<script>
const API = "__API_PREFIX__/status";
const REFRESH = 5;
let countdown = REFRESH;

function cls(status) {
  const s = (status || "").toLowerCase();
  if (["healthy", "ok", "available", "succeeded"].includes(s)) return "ok";
  if (["degraded", "error", "unhealthy", "failed"].includes(s)) return "err";
  return "warn";
}
function fmtUptime(s) {
  s = Math.floor(s); const d = Math.floor(s / 86400); s %= 86400;
  const h = Math.floor(s / 3600); s %= 3600; const m = Math.floor(s / 60); s %= 60;
  return (d ? d + "d " : "") + (h ? h + "h " : "") + (m ? m + "m " : "") + s + "s";
}
function badge(status) {
  return `<span class="badge ${cls(status)}"><span class="dot ${cls(status)}"></span>${status || "unknown"}</span>`;
}

async function refresh() {
  let d;
  try { d = await (await fetch(API, {cache: "no-store"})).json(); }
  catch (e) { document.getElementById("updated").textContent = "unreachable"; return; }

  document.getElementById("title").textContent = d.platform.name;
  document.getElementById("version").textContent = "v" + d.platform.version;
  document.getElementById("uptime").textContent = fmtUptime(d.platform.uptime_seconds);
  document.getElementById("updated").textContent = new Date().toLocaleTimeString();
  document.getElementById("overall-dot").className = "dot " + cls(d.status);

  // Summary cards
  const cards = Object.entries(d.summary).map(([k, v]) => {
    const bad = v.total - v.healthy;
    return `<div class="card"><div class="n">${v.total}</div><div class="l">${k}</div>
      <div class="sub ${bad ? '' : 'muted'}" style="color:${bad ? 'var(--warn)' : ''}">
      ${v.healthy}/${v.total} healthy</div></div>`;
  }).join("");
  document.getElementById("cards").innerHTML = cards;

  // Tasks
  const t = d.tasks;
  if (!t.enabled) {
    document.getElementById("tasks").innerHTML = `<div class="empty">Task system disabled.</div>`;
  } else if (t.error) {
    document.getElementById("tasks").innerHTML = `<div class="empty">Error: ${t.error}</div>`;
  } else {
    const rows = Object.entries(t.by_status || {})
      .map(([s, n]) => `<tr><td>${badge(s)}</td><td>${n}</td></tr>`).join("");
    document.getElementById("tasks").innerHTML = `<table>
      <tr><th>status</th><th>count</th></tr>${rows}
      <tr><td>running now</td><td>${t.running} / ${t.max_concurrency}</td></tr>
      <tr><td>total</td><td>${t.total}</td></tr>
      </table><div class="muted" style="margin-top:8px">targets:
      ${(t.supported_targets || []).map(x => `<code>${x}</code>`).join(" ")}</div>`;
  }

  // Storage
  document.getElementById("storage").innerHTML = `<table>
    <tr><th>field</th><th>value</th></tr>
    ${Object.entries(d.storage).map(([k, v]) =>
      `<tr><td>${k}</td><td>${k === 'status' ? badge(v) : JSON.stringify(v)}</td></tr>`).join("")}
    </table>`;

  // Resource sections
  const sections = Object.entries(d.resources).map(([name, sec]) => {
    const items = Object.entries(sec.items);
    const body = items.length
      ? `<table><tr><th>name</th><th>status</th><th>details</th></tr>` +
        items.map(([n, info]) => {
          const extra = Object.entries(info)
            .filter(([k]) => k !== "status")
            .map(([k, v]) => `${k}=${v}`).join(", ");
          return `<tr><td><b>${n}</b></td><td>${badge(info.status)}</td>
            <td class="muted">${extra}</td></tr>`;
        }).join("") + `</table>`
      : `<div class="empty">None registered.</div>`;
    return `<section><h2>${name} (${sec.healthy}/${sec.total})</h2>${body}</section>`;
  }).join("");
  document.getElementById("sections").innerHTML = sections;

  countdown = REFRESH;
}

setInterval(() => {
  countdown -= 1;
  document.getElementById("countdown").textContent = "refresh in " + Math.max(countdown, 0) + "s";
  if (countdown <= 0) refresh();
}, 1000);
refresh();
</script>
</body>
</html>
"""
