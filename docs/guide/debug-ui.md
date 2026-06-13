# Debug UI

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Interactive Agent Playground</h3>
</div>

---

Agentomatic includes a built-in ChatGPT-like conversational playground. Powered by **Chainlit**, it provides an interactive console to inspect, test, and debug your agents without writing any frontend code.

---

## 🏁 Installation & Launch

Install the web extra components:

```bash
pip install agentomatic[ui]
```

You can launch the Debug UI in two modes:

### 1. Embedded Mode (Recommended)
Mounts the Chainlit interface directly into your FastAPI web application. 

```bash
agentomatic run --with-ui
```

This runs the platform and maps the UI to `/chat` (e.g., `http://localhost:8000/chat`). API requests and chat sessions utilize the same persistence backend and middleware filters.

### 2. Standalone Mode
Launches the chat console directly, binding to a custom port.

```bash
agentomatic ui --port 8000
```

---

## 🎨 Interface Features & Layout

The Debug UI is fully optimized with a custom dark theme matching Agentomatic's sleek aesthetics:

### 🔍 1. Agent Selector
A top-navigation dropdown lists all registered agents discovered by your platform registry. Select an agent to dynamically load its input form and instructions.

### 🏷️ 2. Prompt Version Selector
Inspect and switch prompt versions (e.g., `v1`, `v2`, `v1_formal`) on the fly. Chat queries are executed against the selected version, letting you manually test and compare prompt behaviors.

### 🌊 3. Token-by-Token Streaming
If your agent supports streaming (via `__init__.py` manifest metadata and SSE), response completions stream onto the screen in real-time, matching modern chat experiences.

### 🛠️ 4. Expandable Tool Call Visualizations
Intermediate agent actions, like tool calls, are captured and rendered as clean, expandable panels in the chat flow. Click on a panel to inspect the exact input arguments and JSON output returned by the tool.

### 🧠 5. Chain-of-Thought & Reasoning Spans
If your graph returns reasoning or step-by-step logs, Agentomatic highlights these in collapsible cards, showing the agent's thought process before the final answer is rendered.

### 📖 6. Citations & Sources
Citations returned by RAG pipelines are rendered as clickable footers or badges at the bottom of messages, referencing PDFs, web links, or documentation files.

### 💬 7. User Feedback Collection
Every response includes thumbs-up and thumbs-down icons. Users can submit rating scores and commentary directly from the UI, which are immediately saved to the platform's SQL/Memory database and exported for prompt optimization datasets.

---

## ⚙️ Advanced Customization

You can customize Chainlit's configuration (colors, system messages, layout) by dropping a `.chainlit/` configuration directory into your project.

### Creating custom config files
When running `agentomatic run --with-ui` for the first time, Agentomatic generates a default `.chainlit/config.toml` file. You can edit this file to customize:

- **Theme colors**: Adjust primary, background, and paper colors.
- **User Interface layout**: Enable/disable chat history logs or file uploads.
- **Intro message**: Customize the header title, logo, and quickstart text boxes shown when a user opens a new thread.

Example edit in `.chainlit/config.toml`:
```toml
[theme]
# Customize colors to match your brand style
primary = "#d53f8c"
background = "#1a202c"
font_family = "Inter, sans-serif"

[UI]
name = "My Enterprise AI Assistant"
show_readme = false
```
