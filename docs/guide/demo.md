# Demo Command

The `agentomatic demo` command launches a fully-functional platform with a built-in demo agent and Agentomatic Studio enabled — perfect for quick testing and exploration.

## Quick Start

```bash
agentomatic demo
```

This starts:

- **API server** at `http://localhost:8000`
- **Agentomatic Studio** at `http://localhost:8000/studio/ui/`
- **Demo agent** (`demo_assistant`) pre-loaded and ready to query

!!! tip "No agents folder required"
    The `demo` command creates a temporary agent in-memory. You don't need any project setup — just install agentomatic and run.

## What's Included

The demo agent demonstrates several Studio features:

| Feature | Demo Coverage |
|---|---|
| Graph Visualization | Custom 5-node graph via `@studio_graph` |
| SSE Streaming | Real-time node transitions with timing |
| State Inspection | Per-thread input/output tracking |
| Execution History | Multi-step trace recording |
| Decorators | `@studio_graph` and `@studio_state` examples |

## Options

```bash
agentomatic demo --host 0.0.0.0 --port 8000
```

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | `8000` | Port to listen on |

## Use Cases

- **First-time exploration**: See what Agentomatic Studio looks like before building your own agents
- **CI/CD smoke tests**: Verify the platform starts correctly in your pipeline
- **Frontend development**: Test the Studio React frontend against a real backend
- **Workshop demos**: Quickly show off the platform to teammates or at conferences

## What Happens

1. A temporary agent directory is created with a demo agent
2. The agent uses `@studio_graph` to declare a custom graph: `Input → Research → Analyze → Synthesize → Respond → Output`
3. The platform starts with `--studio` enabled
4. Studio is available immediately at `/studio/ui/`

!!! info "Next Steps"
    After exploring the demo, create your own agents with `agentomatic init my_agent` and run with `agentomatic run --studio`.
