# Scaffolding Templates

Agentomatic ships with 5 templates for quick agent creation.

## Usage

```bash
# Interactive (with questionary installed)
agentomatic init my_agent

# Non-interactive
agentomatic init my_agent --template basic
```

## Available Templates

### `basic` — Minimal Agent

```bash
agentomatic init my_agent --template basic
```

7 files. Best for quick prototyping.

### `full` — All Override Files

```bash
agentomatic init my_agent --template full
```

11 files. Includes config, schemas, tools, custom api router.

### `rag` — Retrieval-Augmented Generation

```bash
agentomatic init knowledge_bot --template rag
```

9 files. Two-stage pipeline: retrieve → generate.

### `chatbot` — Conversational Agent

```bash
agentomatic init assistant --template chatbot
```

8 files. Conversation-aware with memory support.

### `custom` — Framework-Agnostic

```bash
agentomatic init simple --template custom
```

4 files. No LangGraph dependency — pure Python.
