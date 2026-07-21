# Agentomatic + Studio — SCOOPER Gemini validation

## Goal
Use the SCOOPER Gemini stack (`:18765`) to extensively test Agentomatic Studio
(debug, logs, chat, control plane, pipelines, plugins).

## Status (in progress)
- [x] Studio defaults to same-origin when embedded under `/studio/ui`
- [x] Remove obsolete bare `/agents` fallbacks → `/studio/agents` + `/api/v1/agents`
- [x] Client request timeouts (avoid infinite "Checking server connectivity…")
- [x] Normalize thread/message API envelopes (`{threads}`, `{messages}`, `thread_id`)
- [x] Timed `/health` probes + `/ready` alias in agentomatic
- [x] Control plane skips `/studio` + probes; SCOOPER enables control plane by default
- [ ] Rebuild Studio UI into `agentomatic/studio/static` (`./scripts/build_studio.sh`)
- [ ] Restart Docker Desktop / `ai_platform` (daemon was returning 500 / hang)
- [ ] E2E verify: connect, graph, chat/stream, threads, control, pipelines, plugins
- [ ] Commit + push `agentomatic` and `agentomatic-studio` with clean conventional messages
- [ ] History rewrite for polluted commit subjects — **needs explicit force-push approval**

## Local verify
```bash
# After Docker Desktop is healthy:
cd SCOOPER_NEW && docker compose up -d --force-recreate ai_platform
curl -sS http://127.0.0.1:18765/readiness
curl -sS http://127.0.0.1:18765/studio/info
curl -sS http://127.0.0.1:18765/api/v1/control

# Bundle latest Studio into agentomatic:
cd agentomatic && ./scripts/build_studio.sh ../agentomatic-studio
```
