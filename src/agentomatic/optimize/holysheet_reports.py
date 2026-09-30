"""HolySheet dashboard builders for fit / eval reports.

HolySheet ``Section`` / ``Tabs`` only render content that is nested in
``children``. Flat ``report.add(Section)`` followed by sibling blocks produces
empty section cards — these builders always nest content correctly.

HolySheet's ``Accordion`` block is not rendered by its bundled viewer (panels
are silently dropped), so panels are flattened with :func:`_panel_blocks`.
"""

from __future__ import annotations

import difflib
import json
import textwrap
from typing import Any

_DIFF_WIDTH = 100


def _display_lines(text: str, width: int = _DIFF_WIDTH) -> list[str]:
    """Split a prompt into lines wrapped at ``_DIFF_WIDTH`` for readable diffs.

    Prompts are often one long line; a unified diff of that is a single
    unreadable ``-``/``+`` pair. Wrapping first makes the diff show the
    sentences that actually changed.
    """
    lines: list[str] = []
    for raw in (text or "").splitlines():
        lines.extend(textwrap.wrap(raw, width, break_on_hyphens=False) or [""])
    return lines


def _unified_diff(before: str, after: str, *, a: str = "before", b: str = "after") -> str:
    """Unified diff of two prompts on wrapped lines ('' when identical)."""
    if (before or "") == (after or ""):
        return ""
    return "\n".join(
        difflib.unified_diff(
            _display_lines(before), _display_lines(after), fromfile=a, tofile=b, lineterm=""
        )
    )


def _wrapped(text: str, width: int = _DIFF_WIDTH) -> str:
    """Prompt text wrapped for a code block (long lines would be cut off)."""
    return "\n".join(_display_lines(text, width))


def _panel_blocks(panels: list[dict[str, Any]]) -> list[Any]:
    """Flatten accordion-style panels into always-visible blocks.

    Each panel becomes a Markdown heading (title + subtitle) followed by its
    children.
    """
    from holysheet import Markdown

    blocks: list[Any] = []
    for panel in panels:
        subtitle = panel.get("subtitle")
        heading = f"#### {panel.get('title', '')}"
        if subtitle:
            heading += f"\n\n_{subtitle}_"
        blocks.append(Markdown(content=heading))
        blocks.extend(panel.get("children") or [])
    return blocks


def _safe_json(value: Any, *, limit: int | None = None) -> str:
    """Pretty-print JSON (or stringify) for report panels."""
    try:
        text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    if limit is not None and len(text) > limit:
        return text[:limit] + "\n…(truncated)"
    return text


def _info_items(pairs: list[tuple[str, Any]]) -> list[dict[str, str]]:
    """Build InfoList items, skipping empty values."""
    items: list[dict[str, str]] = []
    for key, value in pairs:
        if value is None or value == "" or value == [] or value == {}:
            continue
        if isinstance(value, (list, tuple)):
            rendered = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            rendered = ", ".join(f"{k}={v}" for k, v in value.items())
        else:
            rendered = str(value)
        if rendered:
            items.append({"key": key, "value": rendered})
    return items


def _deployment_blocks(rec: Any) -> list[Any]:
    """Render deployment recommendation as KPI + InfoList + notes."""
    from holysheet import KPI, Callout, Columns, InfoList, Markdown

    if rec is None:
        return []
    if not isinstance(rec, dict):
        # dataclass-like
        data = {
            "prompt_version": getattr(rec, "prompt_version", None),
            "confidence": getattr(rec, "confidence", None),
            "expected_improvement": getattr(rec, "expected_improvement", None),
            "baseline_score": getattr(rec, "baseline_score", None),
            "projected_score": getattr(rec, "projected_score", None),
            "model_params": getattr(rec, "model_params", None),
            "monitoring": getattr(rec, "monitoring", None),
            "safety_notes": getattr(rec, "safety_notes", None),
            "deployment_recommendation": getattr(rec, "rollout", None)
            or getattr(rec, "deployment_recommendation", None),
            "rollout": getattr(rec, "rollout", None),
        }
    else:
        data = dict(rec)

    nested = data.get("deployment_recommendation")
    rollout = data.get("rollout") or nested or {}
    if not isinstance(rollout, dict) and callable(getattr(rollout, "to_dict", None)):
        rollout = rollout.to_dict()  # RolloutConfig is a slots dataclass (no __dict__)
    if hasattr(rollout, "__dict__") and not isinstance(rollout, dict):
        rollout = {
            "strategy": getattr(rollout, "strategy", None),
            "rollout": getattr(rollout, "strategy", None),
            "weight": getattr(rollout, "initial_weight", getattr(rollout, "weight", None)),
            "initial_weight": getattr(rollout, "initial_weight", None),
            "monitoring_hours": getattr(rollout, "monitoring_hours", None),
        }
    if not isinstance(rollout, dict):
        rollout = {}

    strategy = rollout.get("strategy") or rollout.get("rollout") or data.get("strategy") or "—"
    weight = rollout.get("weight", rollout.get("initial_weight", None))
    monitor_h = rollout.get("monitoring_hours", "—")
    confidence = str(data.get("confidence") or "—")
    conf_status = (
        "positive"
        if confidence == "high"
        else ("negative" if confidence in {"low", "no_improvement"} else "neutral")
    )

    blocks: list[Any] = [
        Columns(
            layout="equal",
            children=[
                KPI(label="Version", value=str(data.get("prompt_version") or "—")),
                KPI(label="Confidence", value=confidence, status=conf_status),
                KPI(
                    label="Rollout",
                    value=f"{strategy}"
                    + (f" @ {float(weight):.0%}" if weight is not None else ""),
                ),
                KPI(label="Monitor", value=f"{monitor_h}h"),
            ],
        )
    ]
    info = _info_items(
        [
            ("Expected improvement", data.get("expected_improvement")),
            ("Baseline score", data.get("baseline_score")),
            ("Projected score", data.get("projected_score")),
            ("Model params", data.get("model_params")),
        ]
    )
    monitoring = data.get("monitoring") or {}
    if isinstance(monitoring, dict) and monitoring:
        info.extend(
            _info_items(
                [
                    ("Monitor metrics", monitoring.get("metrics")),
                    ("Rollback threshold", monitoring.get("rollback_threshold")),
                    ("Rollback instructions", monitoring.get("rollback_instructions")),
                ]
            )
        )
    if info:
        blocks.append(InfoList(title="Deployment details", items=info))

    notes = list(data.get("safety_notes") or [])
    if notes:
        blocks.append(
            Callout(
                content="\n".join(f"- {n}" for n in notes),
                variant="highlight",
            )
        )
    elif confidence == "no_improvement":
        blocks.append(Markdown(content="_No rollout recommended — fit did not beat baseline._"))
    return blocks


def _prompt_evolution_entries(
    *,
    baseline_prompt: str,
    baseline_score: float,
    prompt_history: list[Any],
) -> list[dict[str, Any]]:
    """Build chronological prompt versions with unified diffs and score deltas."""
    prev = baseline_prompt or ""
    prev_score = float(baseline_score)
    versions: list[dict[str, Any]] = [
        {
            "version": 0,
            "score": prev_score,
            "delta": 0.0,
            "accepted": True,
            "candidate": "baseline",
            "prompt": prev,
            "diff": "",
        }
    ]
    for entry in prompt_history:
        if not isinstance(entry, dict):
            continue
        snap = str(entry.get("prompt_snapshot") or "")
        score = float(entry.get("score") or 0.0)
        accepted = bool(entry.get("accepted"))
        new_prompt = snap or prev
        diff_text = _unified_diff(
            prev, new_prompt, a=f"v{len(versions) - 1}", b=f"v{len(versions)}"
        )
        versions.append(
            {
                "version": len(versions),
                "score": score,
                "delta": score - prev_score,
                "accepted": accepted,
                "candidate": str(entry.get("candidate_name") or ""),
                "prompt": new_prompt,
                "diff": diff_text,
                "what_worked": list(entry.get("what_worked") or []),
                "what_failed": list(entry.get("what_failed") or []),
                "next_focus": list(entry.get("next_focus") or []),
                "judge_insights": list(entry.get("judge_insights") or []),
            }
        )
        if accepted or new_prompt != prev:
            prev = new_prompt
        prev_score = score
    return versions


def build_fit_holysheet_report(
    result: Any,
    output_path: Any,
    *,
    keras_history: dict[str, list[float]] | None = None,
    eval_scores: dict[str, float] | None = None,
    dataset_sizes: dict[str, int] | None = None,
    optimizer_name: str = "",
    stack_name: str = "",
    model_name: str = "",
    run_config: dict[str, Any] | None = None,
    epochs: list[Any] | None = None,
    baseline_eval: Any = None,
    final_eval: Any = None,
    eval_dataset: Any = None,
    baseline_eval_scores: dict[str, float] | None = None,
    dataset_stats: dict[str, Any] | None = None,
) -> str:
    """Interactive HolySheet dashboard for a (merged, multi-epoch) PromptFitResult.

    Section order: verdict, test scoreboard, what changed in the prompt,
    epochs, all candidates, examples before vs after, data & settings (see
    :func:`training_sections`), then run configuration, recommendations,
    curves, prompt-evolution learnings and failure analysis.
    """
    from holysheet import (
        KPI,
        CodeBlock,
        Columns,
        DataTable,
        InfoList,
        LineChart,
        Markdown,
        Report,
        Section,
        Tabs,
    )

    keras_history = keras_history or {}
    eval_scores = eval_scores or {}
    dataset_sizes = dataset_sizes or {}
    run_config = run_config or {}

    opt_name = (
        optimizer_name
        or getattr(result, "optimizer_name", "")
        or run_config.get("optimizer")
        or ""
    )
    sizes = dataset_sizes or getattr(result, "dataset_sizes", None) or {}
    early_stop = getattr(result, "early_stop_reason", None) or ""

    meta_bits = [
        f"agent=`{result.agent}`",
        f"experiment=`{result.experiment_id}`",
    ]
    if stack_name:
        meta_bits.append(f"stack=`{stack_name}`")
    if model_name:
        meta_bits.append(f"model=`{model_name}`")
    if opt_name:
        meta_bits.append(f"optimizer=`{opt_name}`")

    report = Report(
        title="PromptFitter Report",
        subtitle=" · ".join(meta_bits),
        theme="dark",
        author="agentomatic",
    )

    for block in training_sections(
        result,
        epochs=list(epochs or [result]),
        baseline_eval=baseline_eval,
        final_eval=final_eval,
        eval_dataset=eval_dataset,
        baseline_eval_scores=baseline_eval_scores,
        eval_scores=eval_scores,
        dataset_stats=dataset_stats,
        dataset_sizes=dataset_sizes,
    ):
        report.add(block)

    # ── Run configuration ──────────────────────────────────────────────
    cfg_items = _info_items(
        [
            ("Stack", stack_name or run_config.get("stack")),
            ("Model", model_name or run_config.get("model")),
            ("Optimizer", opt_name),
            ("Epochs", run_config.get("epochs")),
            ("Max trials", run_config.get("max_trials") or run_config.get("trials")),
            ("Patience", run_config.get("patience")),
            ("Augment", run_config.get("augment")),
            ("Required keys", run_config.get("required_keys")),
            ("Judge dimensions", run_config.get("judge_dimensions")),
            ("Judge weight", run_config.get("judge_weight")),
            ("Min abs. improvement", run_config.get("min_absolute_improvement")),
            ("Early-stop / stop reason", early_stop or "completed normally"),
            ("Dataset sizes", sizes),
        ]
    )
    cfg_children: list[Any] = []
    if cfg_items:
        cfg_children.append(InfoList(title="Run settings", items=cfg_items))
    criteria = run_config.get("judge_criteria")
    if criteria:
        cfg_children.append(Markdown(content=f"**Judge criteria**\n\n{criteria}"))
    if not cfg_children:
        cfg_children.append(Markdown(content="_No run configuration captured._"))
    report.add(
        Section(
            title="Run Configuration",
            description="Stack, optimizer budget, dataset, and judge setup",
            children=cfg_children,
        )
    )

    # ── Recommendations ────────────────────────────────────────────────
    rec_children: list[Any] = []
    suggestions = list(result.suggestions or [])
    if suggestions:
        rec_children.append(Markdown(content="\n".join(f"- {s}" for s in suggestions[:12])))
    deployment = getattr(result, "deployment_recommendation", None)
    if deployment:
        rec_children.append(Markdown(content="### Deployment recommendation"))
        rec_children.extend(_deployment_blocks(deployment))
    param_suggestions = getattr(result, "param_suggestions", None) or {}
    if param_suggestions:
        rows = []
        for name, delta in param_suggestions.items():
            if isinstance(delta, dict):
                rows.append(
                    {
                        "param": name,
                        "old": str(delta.get("old_value", ""))[:120],
                        "new": str(delta.get("new_value", ""))[:120],
                        "reason": str(delta.get("reason", ""))[:200],
                    }
                )
            else:
                rows.append(
                    {
                        "param": name,
                        "old": str(getattr(delta, "old_value", ""))[:120],
                        "new": str(getattr(delta, "new_value", ""))[:120],
                        "reason": str(getattr(delta, "reason", ""))[:200],
                    }
                )
        if rows:
            rec_children.append(
                DataTable(
                    title="Parameter changes",
                    data=rows,
                    columns=["param", "old", "new", "reason"],
                )
            )
    metric_deltas = getattr(result, "metric_deltas", None) or {}
    if metric_deltas:
        rec_children.append(
            DataTable(
                title="Metric deltas",
                data=[
                    {"metric": k, "delta": round(float(v), 4)} for k, v in metric_deltas.items()
                ],
                columns=["metric", "delta"],
            )
        )
    if not rec_children:
        rec_children.append(Markdown(content="_No recommendations for this run._"))
    report.add(
        Section(
            title="Recommendations",
            description="Suggested prompt / rollout actions from the fit",
            children=rec_children,
        )
    )

    # ── Curves & metrics tabs ──────────────────────────────────────────
    curve_blocks: list[Any] = []
    scores = list(getattr(result, "score_history", None) or getattr(result, "history", []) or [])
    if scores:
        curve = [
            {
                "round": i,
                "best_score": round(float(s), 4),
                "loss": round(1.0 - float(s), 4),
            }
            for i, s in enumerate(scores)
        ]
        curve_blocks.append(
            LineChart(
                title="Best score / loss over fit rounds (incl. baseline seed)",
                data=curve,
                x="round",
                y=["best_score", "loss"],
                height=320,
            )
        )
        curve_blocks.append(
            DataTable(
                title="Fit round scores",
                data=curve,
                columns=["round", "best_score", "loss"],
            )
        )
    else:
        curve_blocks.append(Markdown(content="_No score_history available._"))

    keras_blocks: list[Any] = []
    if keras_history:
        preferred = ("loss", "val_loss", "f1", "val_f1", "judge", "val_judge")
        hist_keys = [k for k in preferred if k in keras_history] + [
            k for k in keras_history if k not in preferred
        ]
        n = max((len(v) for v in keras_history.values()), default=0)
        if n:
            rows = []
            for i in range(n):
                row: dict[str, Any] = {"epoch": i + 1}
                for key in hist_keys:
                    vals = keras_history.get(key)
                    if isinstance(vals, list) and i < len(vals):
                        row[key] = round(float(vals[i]), 4)
                rows.append(row)
            loss_keys = [k for k in ("loss", "val_loss") if k in rows[0]]
            if loss_keys:
                keras_blocks.append(
                    LineChart(
                        title="Train / val loss across epochs",
                        data=rows,
                        x="epoch",
                        y=loss_keys,
                        height=300,
                    )
                )
            score_keys = [k for k in ("f1", "val_f1", "judge", "val_judge") if k in rows[0]]
            if score_keys:
                keras_blocks.append(
                    LineChart(
                        title="Primary scores across epochs",
                        data=rows,
                        x="epoch",
                        y=score_keys,
                        height=300,
                    )
                )
            other_keys = [k for k in hist_keys if k not in set(loss_keys) | set(score_keys)]
            if other_keys:
                # One multi-series chart instead of one empty-looking chart per metric.
                keras_blocks.append(
                    LineChart(
                        title="Other epoch metrics",
                        data=rows,
                        x="epoch",
                        y=other_keys[:8],
                        height=280,
                    )
                )
            keras_blocks.append(
                DataTable(
                    title="Epoch metrics table",
                    data=rows,
                    columns=["epoch", *hist_keys],
                )
            )
    else:
        keras_blocks.append(Markdown(content="_No Keras-style history recorded._"))

    eval_blocks: list[Any] = []
    if eval_scores:
        eval_rows = [
            {"metric": k, "score": round(float(v), 4)} for k, v in sorted(eval_scores.items())
        ]
        eval_blocks.append(
            Columns(
                layout="equal",
                children=[KPI(label=str(r["metric"]), value=r["score"]) for r in eval_rows[:8]],
            )
        )
        eval_blocks.append(
            DataTable(
                title="Held-out evaluate scores",
                data=eval_rows,
                columns=["metric", "score"],
            )
        )
    else:
        eval_blocks.append(Markdown(content="_No held-out evaluate() scores._"))

    report.add(
        Tabs(
            tabs=[
                {"label": "Score / Loss", "children": curve_blocks},
                {"label": "Keras Epochs", "children": keras_blocks},
                {"label": "Held-out Eval", "children": eval_blocks},
            ]
        )
    )

    # ── Prompt evolution ───────────────────────────────────────────────
    prompt_history = list(getattr(result, "prompt_history", None) or [])
    baseline_prompt = getattr(result.baseline_config, "system_prompt", "") or ""
    best_prompt = getattr(result.best_config, "system_prompt", "") or ""
    evolution = _prompt_evolution_entries(
        baseline_prompt=baseline_prompt,
        baseline_score=float(getattr(result, "baseline_score", 0.0) or 0.0),
        prompt_history=prompt_history,
    )

    learn_rows = []
    judge_rows = []
    for entry in prompt_history:
        if not isinstance(entry, dict):
            continue
        epoch = int(entry.get("round_idx", 0)) + 1
        learn_rows.append(
            {
                "epoch": epoch,
                "score": round(float(entry.get("score", 0.0)), 4),
                "accepted": "yes" if entry.get("accepted") else "no",
                "candidate": str(entry.get("candidate_name") or "")[:48],
                "focus": "; ".join(str(x) for x in (entry.get("next_focus") or [])[:4]),
                "failed": "; ".join(str(x) for x in (entry.get("what_failed") or [])[:3]),
                "worked": "; ".join(str(x) for x in (entry.get("what_worked") or [])[:3]),
                "prompt_chars": len(str(entry.get("prompt_snapshot") or "")),
            }
        )
        for insight in entry.get("judge_insights") or []:
            judge_rows.append({"epoch": epoch, "motivation": str(insight)})

    evo_panels = []
    for item in evolution:
        is_best = item["prompt"] == best_prompt and item["version"] > 0
        title = (
            f"v{item['version']} · score {item['score']:.4f} · "
            f"Δ{item['delta']:+.4f} · "
            f"{'ACCEPTED' if item['accepted'] else 'rejected/unchanged'} · "
            f"{item['candidate'] or '—'}"
        )
        if is_best or item["version"] == 0:
            title = ("🏆 " if is_best else "🌱 ") + title
        children: list[Any] = []
        meta_parts = []
        for label, key in (
            ("Focus", "next_focus"),
            ("Worked", "what_worked"),
            ("Failed", "what_failed"),
        ):
            vals = item.get(key) or []
            if vals:
                meta_parts.append(f"**{label}:** " + "; ".join(str(v) for v in vals[:5]))
        insights = item.get("judge_insights") or []
        if insights:
            meta_parts.append("**Judge:** " + " | ".join(str(i) for i in insights[:3]))
        if meta_parts:
            children.append(Markdown(content="\n\n".join(meta_parts)))
        if item["diff"]:
            children.append(
                CodeBlock(code=item["diff"], language="diff", title="Change vs previous version")
            )
        elif item["version"] == 0:
            children.append(
                CodeBlock(code=_wrapped(item["prompt"]) or "(empty)", language="markdown")
            )
        else:
            children.append(Markdown(content="_No text change vs previous version._"))
        evo_panels.append(
            {
                "title": title,
                "subtitle": f"{len(item['prompt'])} chars",
                "children": children,
            }
        )

    prompt_tab_children: list[Any] = []
    if learn_rows:
        prompt_tab_children.append(DataTable(title="Epoch learnings (summary)", data=learn_rows))
    if evo_panels:
        prompt_tab_children.extend(_panel_blocks(evo_panels))
    else:
        prompt_tab_children.append(Markdown(content="_No prompt history._"))
    if judge_rows:
        prompt_tab_children.append(
            DataTable(
                title="Judge samples / motivations",
                data=judge_rows,
                columns=["epoch", "motivation"],
            )
        )

    tabs: list[dict[str, Any]] = [{"label": "Prompt Evolution", "children": prompt_tab_children}]
    few_shot = list(getattr(result.best_config, "few_shot_examples", None) or [])
    if few_shot:
        fs_panels = [
            {
                "title": f"Few-shot #{i}",
                "subtitle": str(ex.get("query", ""))[:80],
                "children": [
                    Markdown(content=f"**Query**\n\n{ex.get('query', '')}"),
                    CodeBlock(
                        code=str(ex.get("response", "")) or "(empty)",
                        language="json",
                        title="Response",
                    ),
                ],
            }
            for i, ex in enumerate(few_shot, 1)
        ]
        tabs.append({"label": "Few-shot examples", "children": _panel_blocks(fs_panels)})

    report.add(Tabs(tabs=tabs))

    # ── Failure analysis (critiques + clusters) ─────────────────────
    failure_children: list[Any] = []
    apo_critiques = [t for t in result.trials or [] if str(t.get("critique") or "").strip()]
    if apo_critiques:
        critique_md = "\n\n".join(
            f"**{t.get('name')}** (epoch {t.get('epoch', 1)}, round {t.get('round')}):\n\n"
            f"{str(t.get('critique'))[:800]}"
            for t in apo_critiques[:8]
        )
        failure_children.append(
            Markdown(content=("### Textual gradients / APO critiques\n\n" + critique_md))
        )
    fc_rows = [
        {
            "label": cluster.get("label", ""),
            "count": cluster.get("count", 0),
            "description": str(cluster.get("description", ""))[:300],
            "fix": str(cluster.get("suggested_fix", ""))[:300],
        }
        for cluster in result.failure_clusters or []
        if isinstance(cluster, dict)
    ]
    if fc_rows:
        failure_children.append(DataTable(title="Failure clusters (train split)", data=fc_rows))
    if failure_children:
        report.add(
            Section(
                title="Failure analysis",
                description="What the optimizer saw going wrong on the reflection (train) data",
                children=failure_children,
            )
        )

    report.export_html(str(output_path))
    return str(output_path)


def _example_judge_text(er: Any) -> str:
    """Extract judge rationale / motivation from ExampleResult.metadata."""
    meta = getattr(er, "metadata", None) or {}
    if not isinstance(meta, dict):
        return ""
    parts: list[str] = []
    # Preferred: per-metric rich blobs from OptimizeMetricAdapter
    for key in ("judge", "local_judge", "llm_judge"):
        blob = meta.get(key)
        if isinstance(blob, dict):
            reason = blob.get("reason") or blob.get("motivation") or ""
            motivation = ""
            inner = blob.get("metadata") or {}
            if isinstance(inner, dict):
                motivation = str(inner.get("motivation") or "")
                hints = inner.get("improvement_hints") or []
                failed = inner.get("what_failed") or []
                worked = inner.get("what_worked") or []
                if worked:
                    parts.append("Worked: " + "; ".join(str(x) for x in worked[:4]))
                if failed:
                    parts.append("Failed: " + "; ".join(str(x) for x in failed[:4]))
                if hints:
                    parts.append("Hints: " + "; ".join(str(x) for x in hints[:4]))
            if motivation:
                parts.append(motivation)
            elif reason:
                parts.append(str(reason))
    if not parts:
        for k, v in meta.items():
            if "reason" in k or "motivation" in k or "feedback" in k:
                if v:
                    parts.append(str(v))
    return "\n".join(parts).strip()


def build_eval_holysheet_report(
    report_obj: Any,
    output_path: Any,
    *,
    stack_name: str = "",
    model_name: str = "",
    split: str = "",
    dataset_sizes: dict[str, int] | None = None,
    run_config: dict[str, Any] | None = None,
) -> str:
    """Interactive HolySheet dashboard for an EvaluationReport."""
    from holysheet import (
        KPI,
        Callout,
        CodeBlock,
        Columns,
        DataTable,
        InfoList,
        Markdown,
        Report,
        Section,
        Tabs,
    )

    dataset_sizes = dataset_sizes or {}
    run_config = run_config or {}
    scores = dict(getattr(report_obj, "scores", {}) or {})
    examples = list(getattr(report_obj, "example_results", []) or [])
    n = len(examples)
    errors = sum(1 for er in examples if getattr(er, "error", None))
    pass_rate = float(getattr(report_obj, "pass_rate", 0.0) or 0.0)
    agent = str(getattr(report_obj, "agent_name", "") or "agent")

    meta_bits = [f"agent=`{agent}`"]
    if stack_name:
        meta_bits.append(f"stack=`{stack_name}`")
    if model_name:
        meta_bits.append(f"model=`{model_name}`")
    if split:
        meta_bits.append(f"split=`{split}`")

    hs = Report(
        title="Agent Evaluation Report",
        subtitle=" · ".join(meta_bits),
        theme="light",
        author="agentomatic",
    )

    summary_kpis: list[Any] = [
        KPI(label="Examples", value=n),
        KPI(label="Pass rate", value=round(pass_rate, 3)),
        KPI(label="Errors", value=errors, status="negative" if errors else "neutral"),
    ]
    for name, score in sorted(scores.items()):
        summary_kpis.append(KPI(label=str(name), value=round(float(score), 4)))
    hs.add(
        Section(
            title="Summary",
            description="Aggregate scores for this evaluation run",
            children=[Columns(layout="equal", children=summary_kpis[:10])],
        )
    )

    cfg_items = _info_items(
        [
            ("Stack", stack_name or run_config.get("stack")),
            ("Model", model_name or run_config.get("model")),
            ("Split", split or run_config.get("split")),
            ("Limit", run_config.get("limit")),
            ("Use judge", run_config.get("use_judge")),
            ("Required keys", run_config.get("required_keys")),
            ("Judge dimensions", run_config.get("judge_dimensions")),
            ("Judge weight", run_config.get("judge_weight")),
            ("Dataset path", run_config.get("dataset_path")),
            ("Dataset sizes", dataset_sizes),
            ("Prefer augmented", run_config.get("prefer_augmented")),
        ]
    )
    cfg_children: list[Any] = []
    if cfg_items:
        cfg_children.append(InfoList(title="Run settings", items=cfg_items))
    criteria = run_config.get("judge_criteria")
    if criteria:
        cfg_children.append(Markdown(content=f"**Judge criteria**\n\n{criteria}"))
    if not cfg_children:
        cfg_children.append(Markdown(content="_No run configuration provided._"))
    hs.add(
        Section(
            title="Run Configuration",
            description="Stack, split, dataset, and judge setup",
            children=cfg_children,
        )
    )

    if scores:
        hs.add(
            Section(
                title="Metrics",
                description="Mean scores across the evaluated split",
                children=[
                    DataTable(
                        title="Aggregate scores",
                        data=[
                            {"metric": k, "score": round(float(v), 4)}
                            for k, v in sorted(scores.items())
                        ],
                        columns=["metric", "score"],
                    )
                ],
            )
        )

    # Per-example table (compact) + accordion (full)
    table_rows: list[dict[str, Any]] = []
    score_keys: dict[str, None] = {}
    panels: list[dict[str, Any]] = []
    rationale_rows = []
    for er in examples:
        er_scores = getattr(er, "scores", {}) or {}
        prediction = getattr(er, "prediction", None) or {}
        eid = str(getattr(er, "example_id", "") or "")
        passed = bool(getattr(er, "passed", False))
        err = getattr(er, "error", None) or ""
        ms = round(float(getattr(er, "duration_ms", 0) or 0), 1)
        rationale = _example_judge_text(er)
        row: dict[str, Any] = {"id": eid, "passed": passed}
        for key, value in er_scores.items():
            score_keys.setdefault(key, None)
            try:
                row[key] = round(float(value), 3)
            except (TypeError, ValueError):
                row[key] = str(value)
        row.update({"error": str(err)[:80], "ms": ms})
        table_rows.append(row)
        if rationale:
            rationale_rows.append({"id": eid, "rationale": rationale})

        panel_children: list[Any] = [
            Markdown(
                content=(
                    f"**Passed:** `{passed}` · **Duration:** `{ms} ms`\n\n"
                    f"**Scores:** `{_safe_json(er_scores)}`"
                )
            )
        ]
        if err:
            panel_children.append(Callout(content=str(err), variant="highlight"))
        if rationale:
            panel_children.append(Markdown(content=f"### Judge rationale\n\n{rationale}"))
        panel_children.append(
            CodeBlock(
                code=_safe_json(prediction) if prediction else "(no prediction)",
                language="json",
                title="Full prediction / output",
            )
        )
        meta = getattr(er, "metadata", None) or {}
        if meta:
            # Strip huge / non-serializable nested objects for display
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, dict):
                    clean_meta[k] = {
                        kk: vv
                        for kk, vv in v.items()
                        if kk != "metric_result" and not str(kk).startswith("_")
                    }
                else:
                    clean_meta[k] = v
            panel_children.append(
                CodeBlock(
                    code=_safe_json(clean_meta),
                    language="json",
                    title="Example metadata",
                )
            )
        panels.append(
            {
                "title": f"{eid} · {'PASS' if passed else 'FAIL'}",
                "subtitle": " · ".join(
                    [*(f"{k}={_fmt(v, 3)}" for k, v in er_scores.items()), f"{ms}ms"]
                ),
                "passed": passed and not err,
                "children": panel_children,
            }
        )

    example_children: list[Any] = []
    if table_rows:
        example_children.append(
            DataTable(
                title="Per-example scores",
                data=[
                    {k: r.get(k, "—") for k in ["id", "passed", *score_keys, "error", "ms"]}
                    for r in table_rows
                ],
                columns=["id", "passed", *score_keys, "error", "ms"],
            )
        )
    if panels:
        # Failures first; details for at most 30 examples (all are in the table).
        shown = sorted(panels, key=lambda panel: bool(panel.get("passed")))[:30]
        example_children.extend(_panel_blocks(shown))
    else:
        example_children.append(Markdown(content="_No examples evaluated._"))

    rationale_children: list[Any] = []
    if rationale_rows:
        rationale_children.append(
            DataTable(
                title="Judge rationales",
                data=rationale_rows,
                columns=["id", "rationale"],
            )
        )
        rationale_children.append(
            Callout(
                content=(
                    "Open the Per-example accordion for full predictions and "
                    "structured judge metadata."
                ),
                variant="note",
            )
        )
    else:
        rationale_children.append(
            Markdown(
                content=(
                    "_No judge rationales captured. Ensure the LLM judge metric "
                    "is enabled and OptimizeMetricAdapter stashes ``last_result``._"
                )
            )
        )

    # Lightweight recommendations from failures
    rec_children: list[Any] = []
    failed = [er for er in examples if not getattr(er, "passed", False)]
    if failed:
        tips = [
            f"- `{getattr(er, 'example_id', '?')}` failed (scores={getattr(er, 'scores', {})})"
            for er in failed[:8]
        ]
        rec_children.append(
            Callout(
                content="Failed examples to inspect:\n" + "\n".join(tips),
                variant="highlight",
            )
        )
    low_metrics = [k for k, v in scores.items() if float(v) < 0.5]
    if low_metrics:
        rec_children.append(
            Markdown(
                content=(
                    "**Low aggregate metrics (< 0.5):** "
                    + ", ".join(f"`{m}`" for m in low_metrics)
                    + "\n\nConsider prompt fit (`train_next.py`) focusing on these dimensions."
                )
            )
        )
    if not rec_children:
        rec_children.append(
            Callout(
                content="All evaluated examples look healthy on the reported metrics.",
                variant="note",
            )
        )

    hs.add(
        Tabs(
            tabs=[
                {"label": "Per-example", "children": example_children},
                {"label": "Judge Rationales", "children": rationale_children},
                {"label": "Recommendations", "children": rec_children},
            ]
        )
    )

    hs.export_html(str(output_path))
    return str(output_path)


# =====================================================================
# Training-report sections (multi-epoch view, candidates, examples, data)
# =====================================================================


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a score (or '—')."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _scores_text(scores: dict[str, Any] | None) -> str:
    """Compact 'metric=score' rendering of a per-example score dict."""
    return " · ".join(f"{k}={_fmt(v, 2)}" for k, v in (scores or {}).items()) or "—"


def training_sections(
    merged: Any,
    *,
    epochs: list[Any],
    baseline_eval: Any = None,
    final_eval: Any = None,
    eval_dataset: Any = None,
    baseline_eval_scores: dict[str, float] | None = None,
    eval_scores: dict[str, float] | None = None,
    dataset_stats: dict[str, Any] | None = None,
    dataset_sizes: dict[str, int] | None = None,
) -> list[Any]:
    """Build the report sections that explain *what the optimization did*.

    Args:
        merged: A ``PromptFitResult`` spanning all epochs (initial baseline →
            final best, every trial tagged with its epoch).
        epochs: The per-epoch ``PromptFitResult`` objects, in order.
        baseline_eval: ``EvaluationReport`` on the untouched test split
            *before* fit (optional).
        final_eval: ``EvaluationReport`` on the same split *after* fit.
        eval_dataset: Examples (``AgentDataset`` or list) to join inputs and
            expected outputs onto the per-example results.
        baseline_eval_scores: Scores-only alternative to ``baseline_eval``.
        eval_scores: Scores-only alternative to ``final_eval``.
        dataset_stats: e.g. ``dataset.metadata["augment_stats"]``.
        dataset_sizes: Split sizes.

    Returns:
        HolySheet blocks, in display order.
    """
    from holysheet import (
        KPI,
        Callout,
        CodeBlock,
        Columns,
        DataTable,
        InfoList,
        Markdown,
        Section,
    )

    blocks: list[Any] = []
    before_scores = dict(getattr(baseline_eval, "scores", None) or baseline_eval_scores or {})
    after_scores = dict(getattr(final_eval, "scores", None) or eval_scores or {})
    initial_prompt = getattr(merged.baseline_config, "system_prompt", "") or ""
    final_prompt = getattr(merged.best_config, "system_prompt", "") or ""
    val_delta = float(merged.best_score) - float(merged.baseline_score)

    # ── Verdict ──────────────────────────────────────────────────────
    kpis = [
        KPI(label="Validation (initial)", value=round(float(merged.baseline_score), 4)),
        KPI(
            label="Validation (final)",
            value=round(float(merged.best_score), 4),
            delta=f"{val_delta:+.4f}",
            status="positive" if val_delta > 0 else ("negative" if val_delta < 0 else None),
        ),
    ]
    if merged.holdout_score is not None:
        hold_before = merged.baseline_holdout_score
        hold_delta = (
            float(merged.holdout_score) - float(hold_before) if hold_before is not None else None
        )
        kpis.append(
            KPI(
                label="Held-out gate",
                value=round(float(merged.holdout_score), 4),
                delta=f"{hold_delta:+.4f}" if hold_delta is not None else None,
            )
        )
    shared = [k for k in after_scores if k in before_scores]
    for key in shared[:3]:
        delta = float(after_scores[key]) - float(before_scores[key])
        kpis.append(
            KPI(
                label=f"Test {key}",
                value=round(float(after_scores[key]), 4),
                delta=f"{delta:+.4f}",
                status="positive" if delta > 0 else ("negative" if delta < 0 else None),
            )
        )
    accepted = [t for t in merged.trials or [] if t.get("decision") == "accepted"]
    verdict_lines = [
        f"- **Prompt {'changed' if final_prompt != initial_prompt else 'unchanged'}** after "
        f"{len(epochs)} epoch(s); {len(accepted)} candidate(s) accepted of "
        f"{sum(1 for t in merged.trials or [] if t.get('phase') == 'minibatch')} scored.",
        f"- Validation {_fmt(merged.baseline_score)} → {_fmt(merged.best_score)} "
        f"(Δ {val_delta:+.4f}).",
    ]
    if merged.holdout_score is not None:
        verdict_lines.append(
            f"- Held-out gate (veto only, never selects): "
            f"{_fmt(merged.baseline_holdout_score)} → {_fmt(merged.holdout_score)}."
        )
    if shared:
        verdict_lines.append(
            "- Untouched test split: "
            + ", ".join(f"{k} {_fmt(before_scores[k])} → {_fmt(after_scores[k])}" for k in shared)
            + "."
        )
    if merged.early_stop_reason:
        verdict_lines.append(f"- Stop reason: {merged.early_stop_reason}.")
    if final_prompt == initial_prompt:
        headline = "Prompt unchanged: no candidate passed the acceptance rules."
    elif shared:
        key = shared[-1]
        headline = (
            f"Prompt improved: validation {_fmt(merged.baseline_score, 3)} → "
            f"{_fmt(merged.best_score, 3)}, test {key} {_fmt(before_scores[key], 3)} → "
            f"{_fmt(after_scores[key], 3)}."
        )
    else:
        headline = (
            f"Prompt changed: validation {_fmt(merged.baseline_score, 3)} → "
            f"{_fmt(merged.best_score, 3)} (evaluate the test split for an unbiased number)."
        )
    blocks.append(
        Section(
            title="Verdict",
            description="Initial vs final — validation (selection), held-out gate, test",
            children=[
                Columns(layout="equal", children=kpis),
                Callout(content=headline, variant="note" if val_delta > 0 else "highlight"),
                Markdown(content="\n".join(verdict_lines)),
            ],
        )
    )

    # ── Scoreboard ───────────────────────────────────────────────────
    if before_scores or after_scores:
        rows = [
            {
                "metric": key,
                "test before": _fmt(before_scores.get(key)),
                "test after": _fmt(after_scores.get(key)),
                "Δ": (
                    f"{float(after_scores[key]) - float(before_scores[key]):+.4f}"
                    if key in before_scores and key in after_scores
                    else "—"
                ),
            }
            for key in sorted(set(before_scores) | set(after_scores))
        ]
        blocks.append(
            Section(
                title="Test scoreboard",
                description="Every compiled metric on the split optimization never saw",
                children=[
                    DataTable(
                        title="Test scores before vs after",
                        data=rows,
                        columns=["metric", "test before", "test after", "Δ"],
                    )
                ],
            )
        )

    # ── What changed in the prompt ───────────────────────────────────
    change_children: list[Any] = []
    diff = _unified_diff(initial_prompt, final_prompt, a="initial", b="final")
    change_children.append(
        CodeBlock(code=diff, language="diff", title="Initial → final prompt")
        if diff
        else Markdown(content="_The prompt did not change: no candidate passed acceptance._")
    )
    change_children.append(
        Columns(
            layout="equal",
            children=[
                CodeBlock(
                    code=_wrapped(initial_prompt, 64) or "(empty)",
                    language="markdown",
                    title=f"Initial prompt ({len(initial_prompt)} chars)",
                ),
                CodeBlock(
                    code=_wrapped(final_prompt, 64) or "(empty)",
                    language="markdown",
                    title=f"Final prompt ({len(final_prompt)} chars)",
                ),
            ],
        )
    )
    timeline = []
    previous = initial_prompt
    for trial in accepted:
        prompt = str(trial.get("system_prompt") or "")
        timeline.append(
            {
                "title": (
                    f"Epoch {trial.get('epoch', 1)} · round {trial.get('round', '—')} · "
                    f"{trial.get('name', '')} · validation {_fmt(trial.get('incumbent_score'))} → "
                    f"{_fmt(trial.get('score'))} · held-out {_fmt(trial.get('holdout_score'))}"
                ),
                "children": [
                    Markdown(content=f"**Why accepted:** {trial.get('reason') or '—'}"),
                    CodeBlock(
                        code=_unified_diff(previous, prompt, a="previous", b="accepted")
                        or "(no text change — parameters / few-shot only)",
                        language="diff",
                        title="Change vs previous best",
                    ),
                ],
            }
        )
        previous = prompt or previous
    if timeline:
        change_children.append(Markdown(content="### Accepted changes, in order"))
        change_children.extend(_panel_blocks(timeline))
    blocks.append(
        Section(
            title="What changed in the prompt",
            description="Across all epochs: the prompt you started with vs the one you keep",
            children=change_children,
        )
    )

    # ── Epochs ───────────────────────────────────────────────────────
    if len(epochs) > 1:
        blocks.append(
            Section(
                title="Epochs",
                description="Each fit() epoch re-optimizes from the previous best",
                children=[
                    DataTable(
                        title="Per-epoch validation scores",
                        data=[
                            {
                                "epoch": i + 1,
                                "start": _fmt(r.baseline_score),
                                "best": _fmt(r.best_score),
                                "improved": "yes" if r.improved else "no",
                                "held-out": _fmt(r.holdout_score),
                                "candidates": sum(
                                    1 for t in r.trials or [] if t.get("phase") == "minibatch"
                                ),
                                "stop": str(r.early_stop_reason or "")[:120],
                            }
                            for i, r in enumerate(epochs)
                        ],
                    )
                ],
            )
        )

    # ── Candidates ───────────────────────────────────────────────────
    cand_rows = []
    cand_panels: list[dict[str, Any]] = []
    shown_prompts: set[str] = set()
    full_by_name = {
        (t.get("epoch"), t.get("name")): t
        for t in merged.trials or []
        if t.get("phase") == "full_val"
    }
    for t in merged.trials or []:
        if t.get("phase") not in ("minibatch", "skipped"):
            continue
        full = full_by_name.get((t.get("epoch"), t.get("name")), {})
        decision = full.get("decision") or t.get("decision") or ""
        reason = full.get("reason") or t.get("reason") or ""
        cand_rows.append(
            {
                "epoch": t.get("epoch", 1),
                "round": t.get("round", "—"),
                "candidate": str(t.get("name", "")),
                "source": str(t.get("source", "")),
                "minibatch": _fmt(t.get("score")),
                "validation": _fmt(full.get("score")),
                "held-out": _fmt(full.get("holdout_score")),
                "confidence": _fmt(full.get("confidence"), 2),
                "decision": decision,
                "reason": str(reason)[:220],
            }
        )
        prompt = str(t.get("system_prompt") or "")
        # One panel per DISTINCT proposed prompt (duplicates are in the table).
        if prompt and prompt not in shown_prompts and len(cand_panels) < 12:
            shown_prompts.add(prompt)
            notes = f"  \n**Notes:** {t.get('mutation_notes')}" if t.get("mutation_notes") else ""
            cand_panels.append(
                {
                    "title": f"{t.get('name')} (epoch {t.get('epoch', 1)}) — "
                    f"{decision or 'screened'}, minibatch {_fmt(t.get('score'))}",
                    "children": [
                        Markdown(
                            content=f"**Decision:** {decision or '—'}  \n**Why:** {reason or '—'}"
                            + notes
                        ),
                        CodeBlock(
                            code=_unified_diff(initial_prompt, prompt, a="initial", b="candidate")
                            or "(same text as the initial prompt)",
                            language="diff",
                            title="Candidate vs initial prompt",
                        ),
                    ],
                }
            )
    cand_children: list[Any] = []
    if cand_rows:
        cand_children.append(
            DataTable(
                title="Candidates",
                data=cand_rows,
                columns=[
                    "epoch",
                    "round",
                    "candidate",
                    "source",
                    "minibatch",
                    "validation",
                    "held-out",
                    "confidence",
                    "decision",
                    "reason",
                ],
            )
        )
        if cand_panels:
            cand_children.extend(_panel_blocks(cand_panels))
    else:
        cand_children.append(Markdown(content="_No candidates were scored._"))
    blocks.append(
        Section(
            title="All candidates",
            description=(
                "Every proposed prompt, its scores and why it was accepted or rejected "
                "(duplicates, not promoted, not significant, did not transfer, …)"
            ),
            children=cand_children,
        )
    )

    # ── Examples: before vs after ────────────────────────────────────
    example_rows = _example_rows(baseline_eval, final_eval, eval_dataset)
    source = "untouched test split"
    if not example_rows:
        source = "validation examples (the selection set)"
        after_by_query = {e.get("query"): e for e in merged.best_examples or []}
        for before in merged.baseline_examples or []:
            after = after_by_query.get(before.get("query"), {})
            example_rows.append(
                {
                    "example": str(before.get("query", ""))[:120],
                    "expected": str(before.get("expected", ""))[:240],
                    "before": str(before.get("response", ""))[:300],
                    "before score": _fmt(before.get("score"), 2),
                    "after": str(after.get("response", ""))[:300],
                    "after score": _fmt(after.get("score"), 2),
                    "Δ": (
                        f"{float(after['score']) - float(before['score']):+.2f}"
                        if "score" in after and "score" in before
                        else "—"
                    ),
                    "why (judge)": str(after.get("feedback") or before.get("feedback") or "")[
                        :300
                    ],
                }
            )
    blocks.append(
        Section(
            title="Examples — before vs after",
            description=f"Per-example answers and scores ({source})",
            children=[
                DataTable(title="Per-example results", data=example_rows)
                if example_rows
                else Markdown(content="_No per-example results available._")
            ],
        )
    )

    # ── Data ─────────────────────────────────────────────────────────
    data_items = _info_items(
        [(f"{k} examples", v) for k, v in (dataset_sizes or {}).items()]
        + [(f"fitter · {k}", v) for k, v in (getattr(merged, "dataset_sizes", None) or {}).items()]
    )
    data_children: list[Any] = []
    if data_items:
        data_children.append(InfoList(title="Splits", items=data_items))
    if dataset_stats:
        by_strategy = dataset_stats.get("by_strategy") or {}
        data_children.append(
            InfoList(
                title="Augmentation",
                items=_info_items(
                    [(k, v) for k, v in dataset_stats.items() if k != "by_strategy"]
                    + [(f"added via {k}", v) for k, v in by_strategy.items()]
                ),
            )
        )
    settings = getattr(merged, "settings", None) or {}
    if settings:
        data_children.append(
            InfoList(title="Fitter settings", items=_info_items(list(settings.items())))
        )
    if data_children:
        blocks.append(
            Section(
                title="Data & settings",
                description="What the optimizer learned from, selected on, and gated with",
                children=data_children,
            )
        )
    return blocks


def _example_rows(baseline_eval: Any, final_eval: Any, eval_dataset: Any) -> list[dict[str, Any]]:
    """Join before/after ``EvaluationReport`` example results by example id."""
    if final_eval is None:
        return []
    examples = getattr(eval_dataset, "examples", eval_dataset) or []
    by_id = {getattr(e, "id", None): e for e in examples}
    before_by_id = {r.example_id: r for r in getattr(baseline_eval, "example_results", None) or []}
    rows: list[dict[str, Any]] = []
    for after in getattr(final_eval, "example_results", None) or []:
        before = before_by_id.get(after.example_id)
        example = by_id.get(after.example_id)
        query = ""
        expected = ""
        if example is not None:
            try:
                point = example.to_datapoint()
                query = point.query
                from agentomatic.optimize.metrics import plain_expected

                expected = plain_expected(point.expected_answer) or ""
            except Exception:  # noqa: BLE001
                query = str(getattr(example, "input", ""))
        after_pred = after.prediction or {}
        before_pred = (before.prediction if before else None) or {}
        before_scores = (before.scores or {}) if before else {}
        deltas = " · ".join(
            f"{k} {float(v) - float(before_scores[k]):+.2f}"
            for k, v in (after.scores or {}).items()
            if k in before_scores
        )
        rationale = ""
        for meta in (after.metadata or {}).values():
            if isinstance(meta, dict) and meta.get("reason"):
                rationale = str(meta["reason"])
                break
        rows.append(
            {
                "example": after.example_id,
                "question": str(query)[:200],
                "expected": str(expected)[:240],
                "before": str(before_pred.get("response", before.error if before else ""))[:300],
                "before scores": _scores_text(before.scores if before else None),
                "after": str(after_pred.get("response", after.error or ""))[:300],
                "after scores": _scores_text(after.scores),
                "Δ": deltas or "—",
                "why (judge)": rationale[:300],
            }
        )
    return rows
