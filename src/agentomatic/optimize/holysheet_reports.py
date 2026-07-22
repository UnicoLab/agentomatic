"""HolySheet dashboard builders for fit / eval reports.

HolySheet ``Section`` / ``Tabs`` / ``Accordion`` only render content that is
nested in ``children`` (or accordion ``panels``). Flat ``report.add(Section)``
followed by sibling blocks produces empty section cards — these builders always
nest content correctly.
"""

from __future__ import annotations

import difflib
import json
from typing import Any


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

    strategy = (
        rollout.get("strategy")
        or rollout.get("rollout")
        or data.get("strategy")
        or "—"
    )
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
        blocks.append(
            Markdown(content="_No rollout recommended — fit did not beat baseline._")
        )
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
        diff_lines = list(
            difflib.unified_diff(
                prev.splitlines(keepends=True),
                new_prompt.splitlines(keepends=True),
                fromfile=f"v{len(versions) - 1}",
                tofile=f"v{len(versions)}",
                lineterm="",
            )
        )
        versions.append(
            {
                "version": len(versions),
                "score": score,
                "delta": score - prev_score,
                "accepted": accepted,
                "candidate": str(entry.get("candidate_name") or ""),
                "prompt": new_prompt,
                "diff": "".join(diff_lines),
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
) -> str:
    """Interactive HolySheet dashboard for a PromptFitResult."""
    from holysheet import (
        KPI,
        Accordion,
        Callout,
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

    improvement = float(result.best_score) - float(result.baseline_score)
    status = "positive" if improvement > 0 else ("neutral" if improvement == 0 else "negative")
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

    # ── Key results ────────────────────────────────────────────────────
    kpis: list[Any] = [
        KPI(label="Baseline Score", value=round(float(result.baseline_score), 4)),
        KPI(
            label="Best Score",
            value=round(float(result.best_score), 4),
            status="positive" if improvement > 0 else None,
        ),
        KPI(
            label="Improvement",
            value=round(improvement, 4),
            delta=f"{improvement:+.4f}",
            status=status,
        ),
        KPI(label="Trials", value=len(result.trials or [])),
        KPI(
            label="Duration",
            value=round(float(result.duration_seconds or 0), 1),
            unit="s",
        ),
    ]
    if result.holdout_score is not None:
        kpis.append(KPI(label="Holdout", value=round(float(result.holdout_score), 4)))
    report.add(
        Section(
            title="Key Results",
            description="Baseline vs best fit outcome",
            children=[Columns(layout="equal", children=kpis)],
        )
    )

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
        cfg_children.append(
            Markdown(content=f"**Judge criteria**\n\n{criteria}")
        )
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
        rec_children.append(
            Callout(
                content="\n".join(f"- {s}" for s in suggestions[:12]),
                variant="note" if improvement > 0 else "highlight",
            )
        )
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
                    {"metric": k, "delta": round(float(v), 4)}
                    for k, v in metric_deltas.items()
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
    scores = list(
        getattr(result, "score_history", None) or getattr(result, "history", []) or []
    )
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
        n = max((len(v) for v in keras_history.values() if isinstance(v, list)), default=0)
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
            score_keys = [
                k
                for k in ("f1", "val_f1", "judge", "val_judge")
                if k in rows[0]
            ]
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
            other_keys = [
                k for k in hist_keys if k not in set(loss_keys) | set(score_keys)
            ]
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
                children=[
                    KPI(label=str(r["metric"]), value=r["score"])
                    for r in eval_rows[:8]
                ],
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
                CodeBlock(code=item["diff"], language="diff", title="Unified diff")
            )
        else:
            children.append(Markdown(content="_No text change vs previous version._"))
        children.append(
            CodeBlock(
                code=item["prompt"] or "(empty prompt)",
                language="markdown",
                title=f"Full prompt ({len(item['prompt'])} chars)",
            )
        )
        evo_panels.append(
            {
                "title": title,
                "subtitle": f"{len(item['prompt'])} chars",
                "default_expanded": item["version"] == 0 or is_best,
                "children": children,
            }
        )

    prompt_tab_children: list[Any] = []
    if learn_rows:
        prompt_tab_children.append(
            DataTable(title="Epoch learnings (summary)", data=learn_rows)
        )
    if evo_panels:
        prompt_tab_children.append(
            Accordion(panels=evo_panels)
        )
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

    # Baseline → best diff + full best prompt
    diff_lines = list(
        difflib.unified_diff(
            baseline_prompt.splitlines(keepends=True),
            best_prompt.splitlines(keepends=True),
            fromfile="baseline",
            tofile="best",
            lineterm="",
        )
    )
    summary_prompt_children: list[Any] = []
    if diff_lines:
        summary_prompt_children.append(
            CodeBlock(code="".join(diff_lines), language="diff", title="Baseline → best")
        )
    else:
        summary_prompt_children.append(
            Markdown(content="_No prompt text change (baseline kept)._")
        )
    summary_prompt_children.append(
        Accordion(
            panels=[
                {
                    "title": f"Full baseline prompt ({len(baseline_prompt)} chars)",
                    "default_expanded": False,
                    "children": [
                        CodeBlock(
                            code=baseline_prompt or "(empty)",
                            language="markdown",
                        )
                    ],
                },
                {
                    "title": f"Full best prompt ({len(best_prompt)} chars)",
                    "default_expanded": True,
                    "children": [
                        CodeBlock(
                            code=best_prompt or "(empty)",
                            language="markdown",
                        )
                    ],
                },
            ]
        )
    )

    few_shot = list(getattr(result.best_config, "few_shot_examples", None) or [])
    if few_shot:
        fs_panels = []
        for i, ex in enumerate(few_shot, 1):
            q = str(ex.get("query", ""))
            r = str(ex.get("response", ""))
            fs_panels.append(
                {
                    "title": f"Few-shot #{i}",
                    "subtitle": q[:80],
                    "default_expanded": i == 1,
                    "children": [
                        Markdown(content=f"**Query**\n\n{q}"),
                        CodeBlock(code=r or "(empty)", language="json", title="Response"),
                    ],
                }
            )
        summary_prompt_children.append(Accordion(panels=fs_panels))

    report.add(
        Tabs(
            tabs=[
                {"label": "Prompt Evolution", "children": prompt_tab_children},
                {"label": "Best Prompt", "children": summary_prompt_children},
            ]
        )
    )

    # ── Trials / failures ──────────────────────────────────────────────
    trial_children: list[Any] = []
    if result.trials:
        trial_rows = [
            {
                "round": t.get("round", "—"),
                "name": str(t.get("name", "")),
                "phase": str(t.get("phase", "")),
                "score": round(float(t.get("score", 0.0)), 4),
                "notes": str(
                    t.get("mutation_notes", "") or t.get("accept_reason", "") or ""
                )[:200],
            }
            for t in result.trials
        ]
        trial_children.append(DataTable(title="Candidates", data=trial_rows))
    else:
        trial_children.append(Markdown(content="_No trials recorded._"))

    if result.failure_clusters:
        fc_rows = []
        for cluster in result.failure_clusters:
            if isinstance(cluster, dict):
                fc_rows.append(
                    {
                        "label": cluster.get("label", ""),
                        "count": cluster.get("count", 0),
                        "description": str(cluster.get("description", ""))[:300],
                        "fix": str(cluster.get("suggested_fix", ""))[:300],
                    }
                )
        if fc_rows:
            trial_children.append(DataTable(title="Failure clusters", data=fc_rows))

    report.add(
        Section(
            title="Trial History",
            description="Candidate prompts evaluated during the fit",
            children=trial_children,
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
        Accordion,
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
    table_rows = []
    panels = []
    rationale_rows = []
    for er in examples:
        er_scores = getattr(er, "scores", {}) or {}
        prediction = getattr(er, "prediction", None) or {}
        eid = str(getattr(er, "example_id", "") or "")
        passed = bool(getattr(er, "passed", False))
        err = getattr(er, "error", None) or ""
        ms = round(float(getattr(er, "duration_ms", 0) or 0), 1)
        rationale = _example_judge_text(er)
        table_rows.append(
            {
                "id": eid,
                "passed": passed,
                "judge": round(float(er_scores.get("judge", 0.0)), 3)
                if "judge" in er_scores
                else "—",
                "f1": round(float(er_scores.get("f1", 0.0)), 3)
                if "f1" in er_scores
                else "—",
                "keywords": round(float(er_scores.get("keywords", 0.0)), 3)
                if "keywords" in er_scores
                else "—",
                "error": str(err)[:80],
                "ms": ms,
            }
        )
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
            panel_children.append(
                Markdown(content=f"### Judge rationale\n\n{rationale}")
            )
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
                "subtitle": f"judge={er_scores.get('judge', '—')} · {ms}ms",
                "default_expanded": (not passed) or bool(err),
                "children": panel_children,
            }
        )

    example_children: list[Any] = []
    if table_rows:
        example_children.append(
            DataTable(
                title="Per-example scores",
                data=table_rows,
                columns=["id", "passed", "judge", "f1", "keywords", "error", "ms"],
            )
        )
    if panels:
        example_children.append(Accordion(panels=panels))
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
            f"- `{getattr(er, 'example_id', '?')}` failed "
            f"(scores={getattr(er, 'scores', {})})"
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
