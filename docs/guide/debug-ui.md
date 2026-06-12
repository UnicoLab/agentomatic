# Debug UI

Agentomatic includes a built-in ChatGPT-like debug interface powered by Chainlit.

## Installation

```bash
pip install agentomatic[ui]
```

## Usage

### Embedded in Platform

```bash
agentomatic run --with-ui
# Open http://localhost:8000/chat
```

### Standalone

```bash
agentomatic ui --port 8000
```

## Features

- **Agent Selector** — Switch between all registered agents
- **Streaming** — Real-time response display
- **Tool Calls** — Expandable panels showing tool invocations
- **Chain of Thought** — Step-by-step reasoning display
- **Citations** — Source attribution display
- **Suggestions** — Clickable follow-up buttons
- **Feedback** — Thumbs up/down stored via platform storage
- **Dark Theme** — Branded agentomatic dark UI
