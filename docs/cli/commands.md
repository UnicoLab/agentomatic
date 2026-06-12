# CLI Commands

## `agentomatic init`

Scaffold a new agent from a template.

```bash
agentomatic init <name> [--template TEMPLATE] [--dir DIR] [--force]
```

| Argument | Description | Default |
|---|---|---|
| `name` | Agent name (snake_case) | Required |
| `--template, -t` | Template: basic, full, rag, chatbot, custom | Interactive |
| `--dir` | Agents directory | `agents` |
| `--force, -f` | Overwrite existing | `false` |

## `agentomatic run`

Start the platform server.

```bash
agentomatic run [--agents-dir DIR] [--host HOST] [--port PORT] [--reload] [--with-ui]
```

| Argument | Description | Default |
|---|---|---|
| `--agents-dir` | Agents directory | `agents` |
| `--host` | Bind address | `0.0.0.0` |
| `--port` | Bind port | `8000` |
| `--reload` | Auto-reload on changes | `false` |
| `--with-ui` | Enable Chainlit debug UI | `false` |

## `agentomatic list`

List discovered agents.

```bash
agentomatic list [--agents-dir DIR]
```

## `agentomatic test`

Interactive agent testing in the terminal.

```bash
agentomatic test <name> [--host HOST] [--port PORT]
```

## `agentomatic inspect`

Show agent structure, manifest, and configuration.

```bash
agentomatic inspect <name> [--agents-dir DIR]
```

## `agentomatic doctor`

Check environment health and dependencies.

```bash
agentomatic doctor [--agents-dir DIR]
```

## `agentomatic ui`

Launch the Chainlit debug UI standalone.

```bash
agentomatic ui [--host HOST] [--port PORT] [--ui-port UI_PORT]
```
