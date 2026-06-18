# Agentomatic Studio 🎨

Agentomatic provides a built-in visual development environment specifically designed to help you debug, inspect, and trace the execution of your agents in real-time. It completely removes the guesswork from debugging multi-agent graphs and provides native support for features like Time-Travel and Conditional Breakpoints.

## Quick Start

You don't need to configure a separate repository or run separate containers. The studio is bundled directly into the `agentomatic` pip package.

To install the optional studio dependencies and launch the server:

```bash
pip install "agentomatic[studio]"
agentomatic run --studio
```

The unified platform will start serving both your API endpoints at `http://localhost:8000` and the Studio UI at `http://localhost:8000/studio/ui/`.

---

## Key Features

### 1. Live Node Streaming
When you execute an agent query via the chat interface, the **Graph View** maps directly to your agent's underlying LangGraph topology. As the execution progresses, nodes will visually pulse and light up. 

- Server-Sent Events (SSE) stream the node transitions directly from the backend `astream_events` API.
- You can follow complex conditional edges and loops perfectly.

### 2. Time-Travel Debugging
Debugging deeply nested loops or non-deterministic agent flows can be difficult. Agentomatic natively records every step using the checkpointer.

- **History View**: Inside the Debug Console, the **Time Travel** tab lists all past checkpoints for the active thread.
- **Replay**: Click **"Replay from here"** on any historical snapshot to branch your thread. The execution will perfectly resume from that prior state, allowing you to try a different input or test a different edge path.

### 3. Conditional Breakpoints
You can freeze execution right before an agent performs a critical action (like calling an external API or making a final decision).

- **Setting Breakpoints**: Right-click any node in the Graph View and select **"Add Breakpoint"**.
- **Execution**: When the graph reaches that node, it will pause. The node will pulse, and the backend checkpointer will suspend the thread.
- **Resuming**: You can either resume execution, or edit the state before continuing.

### 4. Live State Editing
While the graph is suspended (either due to a breakpoint or a `Human-in-the-loop` interrupt), you can manually mutate the internal LangGraph state payload.

- **State View**: In the Debug Console, navigate to the **State** tab to see your `StateGraph` variables in real-time.
- **Editing**: Click **"Edit State"**, modify the JSON directly, and click **"Save"**.
- **Backend Mutate**: The studio fires an update command which natively triggers `graph.aupdate_state(config, values)`. When you resume execution, the agent will use your injected state!

---

## How It Works Under The Hood

The Agentomatic Studio is built on React and communicates via standard Agentomatic APIs mounted under the `/studio/` router.

- **Topology Extraction**: `GET /studio/agents/{name}/graph` extracts your node mapping via `graph.get_graph().to_json()`.
- **SSE Tracking**: `POST /studio/agents/{name}/runs/stream` wraps your execution config with any active breakpoints (`interrupt_before_nodes`). It streams all `on_chat_model_stream`, `on_tool_start`, and custom node events back to the UI.
- **State Patching**: `POST /studio/agents/{name}/threads/{tid}/state` patches the checkpoint database dynamically.

There is no "mock" data — the studio gives you 100% accurate insights into how your code operates in production.
