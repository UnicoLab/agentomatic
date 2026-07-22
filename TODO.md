# Agentomatic + Studio — SCOOPER Gemini validation

## Goal
Use the SCOOPER Gemini stack (`:18765`) to extensively test Agentomatic Studio
(debug, logs, chat, control plane, pipelines, plugins).

## Status
- [x] Studio defaults to same-origin when embedded under `/studio/ui`
- [x] Remove obsolete bare `/agents` fallbacks → `/studio/agents` + `/api/v1/agents`
- [x] Client request timeouts (avoid infinite "Checking server connectivity…")
- [x] Normalize thread/message API envelopes (`{threads}`, `{messages}`, `thread_id`)
- [x] Timed `/health` probes + `/ready` alias in agentomatic
- [x] Control plane skips `/studio` + probes; SCOOPER enables control plane by default
- [x] Connect no longer blocks on graph load; selected agent set on connect
- [x] Backend mounts agent routers under both folder name and slug
- [x] `graph_fn` / get_graph timeouts so Studio Connect cannot wedge the loop
- [x] Rebuild Studio UI into `agentomatic/studio/static` (`./scripts/build_studio.sh`)
- [x] E2E verify: connect, graph, chat, threads, control, pipelines, plugins, connections
- [x] Commit + push `agentomatic` and `agentomatic-studio` with clean conventional messages
- [ ] History rewrite for polluted commit subjects — **needs explicit force-push approval**

## Local verify
```bash
# Native (recommended when Docker Desktop is flaky):
cd SCOOPER_NEW/ai_platform
PYTHONPATH=../../agentomatic/src AGENTOMATIC_ENABLE_STUDIO=1 \
  AGENTOMATIC_ENABLE_CONTROL_PLANE=1 uv run uvicorn main:app --port 18765

curl -sS http://127.0.0.1:18765/readiness
curl -sS http://127.0.0.1:18765/studio/info
curl -sS http://127.0.0.1:18765/api/v1/control
curl -sS http://127.0.0.1:18765/api/v1/agent-assistant/threads

# Bundle latest Studio into agentomatic:
cd agentomatic && ./scripts/build_studio.sh ../agentomatic-studio
```

## Notes
- Local `.env` must use `AI_VECTOR_PROVIDER=local_npz` (not `local_tensorflow`).
  SCOOPER also registers a `local_tensorflow` alias for resilience.
