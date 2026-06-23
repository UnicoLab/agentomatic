# Agent Pipelines

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Compose Multi-Agent Workflows with Zero Boilerplate</h3>
</div>

---

Agent Pipelines allow you to chain multiple agents into complex workflows using **YAML**, **Builder API**, or **Flow decorators**. Pipelines handle orchestration, error handling, and data passing automatically.

---

## 🚀 Quick Start

=== "YAML"

    ```yaml
    # pipeline.yaml
    name: research_pipeline
    description: "Research and summarize"

    steps:
      - name: research
        agent: researcher
      - name: summarize
        agent: summarizer
        condition: "len(ctx.steps.research.output.get('response', '')) > 0"
    ```

=== "Builder API"

    ```python
    from agentomatic.pipelines import PipelineBuilder

    pipeline = (
        PipelineBuilder("research_pipeline")
        .step("research", agent="researcher")
        .step("summarize", agent="summarizer")
        .build()
    )
    ```

=== "Flow Decorators"

    ```python
    from agentomatic.pipelines.flow import flow, step

    @flow(name="research_pipeline")
    class ResearchPipeline:
        @step(agent="researcher")
        def research(self, ctx): ...

        @step(agent="summarizer", after="research")
        def summarize(self, ctx): ...
    ```

---

## 📊 Step Types

| Type | Description | Example |
|------|-------------|---------|
| **Sequential** | Steps run one after another | Default behavior |
| **Parallel** | Multiple agents run concurrently | `type: parallel` |
| **Conditional** | Steps run only if condition is met | `condition: "..."` |
| **Loop** | Steps repeat until condition | `type: loop` |
| **Transform** | Apply a Python function | `type: transform` |

---

## 🔧 Configuration

### Input/Output Contracts

```yaml
input:
  query:
    type: string
    required: true
  context:
    type: object

output:
  response:
    type: string
  sources:
    type: array
```

### Error Handling

```yaml
on_error: continue  # or 'fail_fast'
timeout: 120.0

steps:
  - name: risky_step
    agent: my_agent
    on_error: skip
    retry:
      max_attempts: 3
      delay: 1.0
```

---

## 🧪 Scaffolding

```bash
agentomatic init my_pipeline --template pipeline
```

This creates a `pipeline.yaml` with a sample multi-step workflow ready to customize.

---

## 📖 Further Reading

- [Templates](templates.md) — all available scaffolding templates
- [Class-Based Agents](class-agents.md) — define agents as Python classes
- [Architecture Overview](../architecture/overview.md) — how pipelines fit in
