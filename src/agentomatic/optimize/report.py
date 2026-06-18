"""Report generator for prompt optimization experiments.

Generates interactive HTML dashboard reports via HolySheet when available,
with automatic fallback to a self-contained HTML/SVG report.

Dashboard includes:
- KPI cards (baseline, best, improvement, duration, best iteration)
- Score-vs-iteration line chart (avg + per-metric)
- Full iteration history data table
- Prompt diff (baseline → best) via unified diff
- All prompt versions (readable sections)
"""

from __future__ import annotations

import difflib
import html
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

# =====================================================================
# Public API — signature kept unchanged
# =====================================================================


def generate_html_report(
    result: Any,  # OptimizationResult
    output_path: str | Path | None = None,
) -> str:
    """Generate a self-contained HTML optimization report.

    Uses HolySheet for an interactive dashboard when installed, otherwise
    falls back to the built-in HTML/SVG renderer.

    Args:
        result: OptimizationResult from an optimization run.
        output_path: Path to write the HTML file. Auto-generated if None.

    Returns:
        Path to the generated report file.
    """
    try:
        return _generate_holysheet_report(result, output_path)
    except ImportError:
        logger.debug("holysheet not installed — falling back to built-in HTML report")
        return _generate_fallback_html(result, output_path)


# =====================================================================
# HolySheet report (primary)
# =====================================================================


def _generate_holysheet_report(
    result: Any,
    output_path: str | Path | None = None,
) -> str:
    """Build an interactive dashboard report using HolySheet.

    Raises ``ImportError`` if *holysheet* is not installed so the caller
    can fall back transparently.
    """
    from holysheet import (  # noqa: F811
        KPI,
        DataTable,
        Divider,
        LineChart,
        Markdown,
        Report,
        Section,
    )

    if output_path is None:
        output_path = Path(f".optimize/{result.agent}/report_{result.experiment_id}.html")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── collect metrics ───────────────────────────────────────────────
    iterations = result.history
    metric_names = sorted(
        {m for it in iterations for m in it.per_metric_scores},
    )

    # ── initialise report ─────────────────────────────────────────────
    report = Report(
        title="⚡ Prompt Optimization Report",
        subtitle=f"Agent: {result.agent}  ·  Experiment: {result.experiment_id}",
        theme="dark",
        author="agentomatic optimize",
    )

    # ── 1. KPI cards ─────────────────────────────────────────────────
    report.add(Section(title="Key Results"))

    improvement_pct = f"{result.improvement:+.1f}%"
    improvement_status = "positive" if result.improvement > 0 else "negative"

    report.add(KPI(label="Baseline Score", value=round(result.baseline_score, 4)))
    report.add(
        KPI(
            label="Best Score",
            value=round(result.best_score, 4),
            status="positive",
        )
    )
    report.add(
        KPI(
            label="Improvement",
            value=improvement_pct,
            delta=improvement_pct,
            status=improvement_status,
        )
    )
    report.add(
        KPI(
            label="Duration",
            value=round(result.duration_seconds, 1),
            unit="s",
        )
    )
    report.add(
        KPI(
            label="Best Iteration",
            value=f"#{result.best_iteration}",
        )
    )

    report.add(Divider())

    # ── 2. Score vs Iteration line chart ─────────────────────────────
    report.add(Section(title="📈 Score vs Iteration"))

    chart_data: list[dict[str, Any]] = []
    for it in iterations:
        row: dict[str, Any] = {
            "iteration": "BL" if it.iteration == 0 else str(it.iteration),
            "avg_score": round(it.avg_score, 4),
        }
        for m in metric_names:
            row[m] = round(it.per_metric_scores.get(m, 0.0), 4)
        chart_data.append(row)

    # Primary avg_score line
    report.add(
        LineChart(
            title="Average Score",
            data=chart_data,
            x="iteration",
            y="avg_score",
        )
    )

    # Per-metric lines (one chart per metric keeps things readable)
    if metric_names:
        per_metric_chart_data: list[dict[str, Any]] = []
        for it in iterations:
            row = {
                "iteration": "BL" if it.iteration == 0 else str(it.iteration),
            }
            for m in metric_names:
                row[m] = round(it.per_metric_scores.get(m, 0.0), 4)
            per_metric_chart_data.append(row)

        for m in metric_names:
            report.add(
                LineChart(
                    title=f"Metric: {m}",
                    data=per_metric_chart_data,
                    x="iteration",
                    y=m,
                )
            )

    report.add(Divider())

    # ── 3. Iteration history data table ──────────────────────────────
    report.add(Section(title="📋 Iteration History"))

    table_data: list[dict[str, Any]] = []
    for i, it in enumerate(iterations):
        label = "Baseline" if i == 0 else f"#{it.iteration}"
        delta = f"{it.avg_score - iterations[i - 1].avg_score:+.4f}" if i > 0 else "—"
        n_fail = len(it.failures) if hasattr(it, "failures") else 0

        row_data: dict[str, Any] = {
            "Iteration": label,
            "Avg Score": round(it.avg_score, 4),
            "Δ": delta,
        }
        for m in metric_names:
            row_data[m] = round(it.per_metric_scores.get(m, 0.0), 4)
        row_data["Failures"] = n_fail
        table_data.append(row_data)

    report.add(DataTable(title="Full History", data=table_data))

    report.add(Divider())

    # ── 4. Prompt diff (baseline → best) ─────────────────────────────
    report.add(Section(title="📝 Prompt Evolution"))

    if len(iterations) >= 2:
        baseline = iterations[0]
        best = next(
            (it for it in iterations if it.iteration == result.best_iteration),
            iterations[-1],
        )

        if baseline.prompt != best.prompt:
            diff_text = _unified_diff(
                baseline.prompt,
                best.prompt,
                "Baseline",
                f"Best (#{best.iteration})",
            )
            report.add(
                Markdown(
                    content=(
                        f"### Baseline → Best (#{best.iteration})\n\n```diff\n{diff_text}\n```"
                    ),
                )
            )
        else:
            report.add(Markdown(content="_No prompt changes between baseline and best._"))

        # Last few sequential diffs
        prev_prompt = iterations[0].prompt
        changes = 0
        for it in iterations[1:]:
            if it.prompt != prev_prompt and changes < 3:
                diff_text = _unified_diff(
                    prev_prompt,
                    it.prompt,
                    f"Iteration #{it.iteration - 1}",
                    f"Iteration #{it.iteration}",
                )
                report.add(
                    Markdown(
                        content=(
                            f"### Iteration #{it.iteration - 1} → #{it.iteration}\n\n"
                            f"```diff\n{diff_text}\n```"
                        ),
                    )
                )
                changes += 1
            prev_prompt = it.prompt
    else:
        report.add(Markdown(content="_Only one iteration — no diff available._"))

    report.add(Divider())

    # ── 5. All prompt versions ───────────────────────────────────────
    report.add(Section(title="📄 Full Prompt Versions"))

    for it in iterations:
        is_best = it.iteration == result.best_iteration
        label = "Baseline" if it.iteration == 0 else f"Iteration #{it.iteration}"
        badge = " 🏆" if is_best else ""

        report.add(
            Markdown(
                content=(
                    f"### {label}{badge} — Score: {it.avg_score:.4f}\n\n```\n{it.prompt}\n```"
                ),
            )
        )

    # ── export ────────────────────────────────────────────────────────
    report.export_html(str(output_path))
    logger.info(f"📊 Report saved to {output_path}")
    return str(output_path)


# =====================================================================
# Helpers
# =====================================================================


def _unified_diff(old: str, new: str, old_label: str, new_label: str) -> str:
    """Return a unified-diff string between two prompt texts."""
    diff_lines = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile=old_label,
        tofile=new_label,
        lineterm="",
    )
    return "\n".join(diff_lines)


# =====================================================================
# Fallback: self-contained HTML/SVG report (no external deps)
# =====================================================================


def _generate_fallback_html(
    result: Any,
    output_path: str | Path | None = None,
) -> str:
    """Original built-in HTML report — used when HolySheet is unavailable."""
    if output_path is None:
        output_path = Path(f".optimize/{result.agent}/report_{result.experiment_id}.html")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_content = _build_html(result)
    output_path.write_text(html_content)
    logger.info(f"📊 Report saved to {output_path}")
    return str(output_path)


def _build_html(result: Any) -> str:
    """Build the full fallback HTML report."""
    iterations = result.history
    metric_names = sorted({m for it in iterations for m in it.per_metric_scores})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Optimization Report — {html.escape(result.agent)}</title>
<style>
{_CSS}
</style>
</head>
<body>
<div class="container">

    <!-- Header -->
    <header>
        <h1>⚡ Prompt Optimization Report</h1>
        <div class="meta">
            <span class="badge">Agent: {html.escape(result.agent)}</span>
            <span class="badge">Experiment: {result.experiment_id}</span>
            <span class="badge">Duration: {result.duration_seconds:.1f}s</span>
        </div>
    </header>

    <!-- Summary Cards -->
    <div class="cards">
        <div class="card">
            <div class="card-label">Baseline Score</div>
            <div class="card-value">{result.baseline_score:.4f}</div>
        </div>
        <div class="card card-highlight">
            <div class="card-label">Best Score</div>
            <div class="card-value">{result.best_score:.4f}</div>
        </div>
        <div class="card {"card-green" if result.improvement > 0 else "card-red"}">
            <div class="card-label">Improvement</div>
            <div class="card-value">{result.improvement:+.1f}%</div>
        </div>
        <div class="card">
            <div class="card-label">Best Iteration</div>
            <div class="card-value">#{result.best_iteration}</div>
        </div>
    </div>

    <!-- Metrics Chart -->
    <section>
        <h2>📈 Score vs Iteration</h2>
        <div class="chart-container">
            {_generate_svg_chart(iterations, metric_names)}
        </div>
    </section>

    <!-- Iteration History -->
    <section>
        <h2>📋 Iteration History</h2>
        {_generate_history_table(iterations, metric_names, result.best_iteration)}
    </section>

    <!-- Prompt Diff -->
    <section>
        <h2>📝 Prompt Evolution</h2>
        {_generate_prompt_diffs(iterations, result.best_iteration)}
    </section>

    <!-- All Prompts -->
    <section>
        <h2>📄 Full Prompt Versions</h2>
        {_generate_prompt_accordion(iterations, result.best_iteration)}
    </section>

    <footer>
        Generated by <strong>agentomatic optimize</strong> at {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")}
    </footer>

</div>
</body>
</html>"""


# =====================================================================
# SVG Chart Generator (fallback)
# =====================================================================


def _generate_svg_chart(
    iterations: list[Any],
    metric_names: list[str],
) -> str:
    """Generate an SVG line chart of scores over iterations."""
    if not iterations:
        return "<p>No data</p>"

    width = 800
    height = 300
    padding = 60
    chart_w = width - 2 * padding
    chart_h = height - 2 * padding

    n = len(iterations)
    if n < 2:
        x_step = float(chart_w)
    else:
        x_step = chart_w / (n - 1)

    # Colors for different metrics
    colors = ["#e94560", "#0f3460", "#16c79a", "#f5a623", "#9b59b6", "#3498db"]

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" class="chart">',
        # Background
        f'<rect width="{width}" height="{height}" fill="#1a1a2e" rx="8"/>',
        # Grid lines
    ]

    # Y-axis grid (0.0 to 1.0)
    for i in range(6):
        y_val = i / 5.0
        y_pos = padding + chart_h - (y_val * chart_h)
        svg_parts.append(
            f'<line x1="{padding}" y1="{y_pos}" x2="{padding + chart_w}" y2="{y_pos}" '
            f'stroke="#333" stroke-dasharray="4,4"/>'
        )
        svg_parts.append(
            f'<text x="{padding - 10}" y="{y_pos + 4}" fill="#888" '
            f'text-anchor="end" font-size="11">{y_val:.1f}</text>'
        )

    # X-axis labels
    for i, it in enumerate(iterations):
        x_pos = padding + i * x_step
        label = "BL" if i == 0 else str(it.iteration)
        svg_parts.append(
            f'<text x="{x_pos}" y="{height - 15}" fill="#888" '
            f'text-anchor="middle" font-size="11">{label}</text>'
        )

    # Average score line
    avg_points = []
    for i, it in enumerate(iterations):
        x = padding + i * x_step
        y = padding + chart_h - (it.avg_score * chart_h)
        avg_points.append(f"{x},{y}")

    svg_parts.append(
        f'<polyline points="{" ".join(avg_points)}" fill="none" '
        f'stroke="#e94560" stroke-width="3" stroke-linejoin="round"/>'
    )

    # Data points
    for i, it in enumerate(iterations):
        x = padding + i * x_step
        y = padding + chart_h - (it.avg_score * chart_h)
        svg_parts.append(
            f'<circle cx="{x}" cy="{y}" r="5" fill="#e94560" stroke="#fff" stroke-width="2"/>'
        )
        svg_parts.append(
            f'<text x="{x}" y="{y - 12}" fill="#e94560" text-anchor="middle" '
            f'font-size="10" font-weight="bold">{it.avg_score:.3f}</text>'
        )

    # Per-metric lines
    for m_idx, metric_name in enumerate(metric_names[:5]):  # Max 5 metrics
        color = colors[(m_idx + 1) % len(colors)]
        m_points = []
        for i, it in enumerate(iterations):
            x = padding + i * x_step
            score = it.per_metric_scores.get(metric_name, 0.0)
            y = padding + chart_h - (score * chart_h)
            m_points.append(f"{x},{y}")

        svg_parts.append(
            f'<polyline points="{" ".join(m_points)}" fill="none" '
            f'stroke="{color}" stroke-width="1.5" stroke-dasharray="6,3" '
            f'stroke-linejoin="round" opacity="0.7"/>'
        )

    # Legend
    legend_items = [("avg_score", "#e94560")] + [
        (name, colors[(i + 1) % len(colors)]) for i, name in enumerate(metric_names[:5])
    ]
    for i, (name, color) in enumerate(legend_items):
        lx = padding + 10 + i * 130
        ly = 20
        svg_parts.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="{color}" rx="2"/>')
        svg_parts.append(
            f'<text x="{lx + 18}" y="{ly + 10}" fill="#ccc" font-size="10">{name}</text>'
        )

    # Axis labels
    svg_parts.append(
        f'<text x="{width // 2}" y="{height - 2}" fill="#888" '
        f'text-anchor="middle" font-size="12">Iteration</text>'
    )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


# =====================================================================
# History Table (fallback)
# =====================================================================


def _generate_history_table(
    iterations: list[Any],
    metric_names: list[str],
    best_iteration: int,
) -> str:
    """Generate HTML table of iteration history."""
    header_cols = "".join(f"<th>{html.escape(m)}</th>" for m in metric_names)
    rows = []

    for i, it in enumerate(iterations):
        is_best = it.iteration == best_iteration
        row_class = "best-row" if is_best else ""
        label = "Baseline" if i == 0 else f"#{it.iteration}"
        badge = ' <span class="badge-sm">🏆 BEST</span>' if is_best else ""

        metric_cells = ""
        for m in metric_names:
            score = it.per_metric_scores.get(m, 0.0)
            color = "green" if score >= 0.7 else ("orange" if score >= 0.4 else "red")
            metric_cells += f'<td><span class="score-{color}">{score:.4f}</span></td>'

        # Delta from previous
        if i > 0:
            delta = it.avg_score - iterations[i - 1].avg_score
            delta_class = "delta-pos" if delta >= 0 else "delta-neg"
            delta_str = f'<span class="{delta_class}">{delta:+.4f}</span>'
        else:
            delta_str = "—"

        n_fail = len(it.failures) if hasattr(it, "failures") else 0
        rows.append(
            f'<tr class="{row_class}">'
            f"<td>{label}{badge}</td>"
            f"<td><strong>{it.avg_score:.4f}</strong></td>"
            f"<td>{delta_str}</td>"
            f"{metric_cells}"
            f"<td>{n_fail}</td>"
            f"</tr>"
        )

    return f"""
    <table>
        <thead>
            <tr>
                <th>Iteration</th>
                <th>Avg Score</th>
                <th>Δ</th>
                {header_cols}
                <th>Failures</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    """


# =====================================================================
# Prompt Diff (fallback)
# =====================================================================


def _generate_prompt_diffs(iterations: list[Any], best_iteration: int) -> str:
    """Generate side-by-side prompt diffs between key iterations."""
    if len(iterations) < 2:
        return "<p>Only one iteration — no diff available.</p>"

    diffs = []

    # Baseline → Best
    baseline = iterations[0]
    best = next((it for it in iterations if it.iteration == best_iteration), iterations[-1])

    if baseline.prompt != best.prompt:
        diff_html = _make_diff(
            baseline.prompt, best.prompt, "Baseline", f"Best (#{best.iteration})"
        )
        diffs.append(f"""
        <div class="diff-section">
            <h3>Baseline → Best (#{best.iteration})</h3>
            {diff_html}
        </div>
        """)

    # Sequential diffs (last 3 changes)
    prev_prompt = iterations[0].prompt
    changes = 0
    for it in iterations[1:]:
        if it.prompt != prev_prompt and changes < 3:
            diff_html = _make_diff(
                prev_prompt, it.prompt, f"#{it.iteration - 1}", f"#{it.iteration}"
            )
            diffs.append(f"""
            <div class="diff-section">
                <h3>Iteration #{it.iteration - 1} → #{it.iteration}</h3>
                {diff_html}
            </div>
            """)
            changes += 1
        prev_prompt = it.prompt

    return "\n".join(diffs) if diffs else "<p>No prompt changes detected.</p>"


def _make_diff(old: str, new: str, old_label: str, new_label: str) -> str:
    """Generate HTML diff between two prompts."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()

    differ = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=old_label,
        tofile=new_label,
        lineterm="",
    )

    lines = []
    for line in differ:
        escaped = html.escape(line)
        if line.startswith("+++") or line.startswith("---"):
            lines.append(f'<div class="diff-header">{escaped}</div>')
        elif line.startswith("@@"):
            lines.append(f'<div class="diff-hunk">{escaped}</div>')
        elif line.startswith("+"):
            lines.append(f'<div class="diff-add">{escaped}</div>')
        elif line.startswith("-"):
            lines.append(f'<div class="diff-del">{escaped}</div>')
        else:
            lines.append(f'<div class="diff-ctx">{escaped}</div>')

    return f'<div class="diff-block">{"".join(lines)}</div>'


# =====================================================================
# Prompt Accordion (fallback)
# =====================================================================


def _generate_prompt_accordion(iterations: list[Any], best_iteration: int) -> str:
    """Generate collapsible sections for each prompt version."""
    sections = []
    for it in iterations:
        is_best = it.iteration == best_iteration
        label = "Baseline" if it.iteration == 0 else f"Iteration #{it.iteration}"
        badge = " 🏆" if is_best else ""
        open_attr = "open" if is_best or it.iteration == 0 else ""

        sections.append(f"""
        <details {open_attr}>
            <summary>{label}{badge} — Score: {it.avg_score:.4f}</summary>
            <pre class="prompt-block">{html.escape(it.prompt)}</pre>
        </details>
        """)

    return "\n".join(sections)


# =====================================================================
# CSS (fallback)
# =====================================================================

_CSS = """
:root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --text-dim: #8b949e;
    --accent: #e94560;
    --green: #3fb950;
    --red: #f85149;
    --orange: #d29922;
    --blue: #58a6ff;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
}
.container { max-width: 1100px; margin: 0 auto; padding: 2rem; }
header { text-align: center; margin-bottom: 2rem; }
h1 { font-size: 2rem; margin-bottom: 0.5rem; color: #fff; }
h2 { font-size: 1.3rem; margin: 2rem 0 1rem; color: #fff;
     border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
h3 { font-size: 1.1rem; margin: 1rem 0 0.5rem; color: var(--text-dim); }
.meta { display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap; }
.badge { background: var(--surface); border: 1px solid var(--border);
         border-radius: 12px; padding: 0.25rem 0.75rem; font-size: 0.85rem; color: var(--text-dim); }
.badge-sm { background: var(--accent); color: #fff; border-radius: 4px;
            padding: 0.1rem 0.4rem; font-size: 0.7rem; font-weight: bold; }

/* Cards */
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
         gap: 1rem; margin: 1.5rem 0; }
.card { background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; padding: 1.25rem; text-align: center; }
.card-label { font-size: 0.85rem; color: var(--text-dim); margin-bottom: 0.25rem; }
.card-value { font-size: 1.8rem; font-weight: bold; color: #fff; }
.card-highlight { border-color: var(--accent); }
.card-highlight .card-value { color: var(--accent); }
.card-green .card-value { color: var(--green); }
.card-red .card-value { color: var(--red); }

/* Chart */
.chart-container { background: var(--surface); border-radius: 8px;
                   padding: 1rem; border: 1px solid var(--border); }
.chart { width: 100%; height: auto; }

/* Table */
table { width: 100%; border-collapse: collapse; background: var(--surface);
        border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }
th { background: #1c2333; color: var(--text-dim); padding: 0.75rem;
     text-align: left; font-size: 0.85rem; text-transform: uppercase; }
td { padding: 0.6rem 0.75rem; border-top: 1px solid var(--border); font-size: 0.9rem; }
.best-row { background: rgba(233, 69, 96, 0.08); }
.score-green { color: var(--green); font-weight: bold; }
.score-orange { color: var(--orange); }
.score-red { color: var(--red); }
.delta-pos { color: var(--green); }
.delta-neg { color: var(--red); }

/* Diff */
.diff-section { margin: 1rem 0; }
.diff-block { background: var(--surface); border: 1px solid var(--border);
              border-radius: 8px; padding: 0; overflow: hidden; font-family: monospace;
              font-size: 0.85rem; line-height: 1.5; }
.diff-header { background: #1c2333; padding: 0.3rem 0.75rem; color: var(--text-dim); }
.diff-hunk { background: #1c2333; padding: 0.2rem 0.75rem; color: var(--blue); }
.diff-add { background: rgba(63, 185, 80, 0.12); padding: 0.1rem 0.75rem;
            color: var(--green); border-left: 3px solid var(--green); }
.diff-del { background: rgba(248, 81, 73, 0.12); padding: 0.1rem 0.75rem;
            color: var(--red); border-left: 3px solid var(--red); }
.diff-ctx { padding: 0.1rem 0.75rem; color: var(--text-dim); }

/* Accordion */
details { background: var(--surface); border: 1px solid var(--border);
          border-radius: 8px; margin: 0.5rem 0; }
summary { padding: 0.75rem 1rem; cursor: pointer; font-weight: 500;
          color: var(--text); user-select: none; }
summary:hover { background: rgba(255,255,255,0.03); }
.prompt-block { padding: 1rem; background: #0d1117; overflow-x: auto;
                white-space: pre-wrap; font-size: 0.85rem; line-height: 1.6;
                color: var(--text-dim); border-top: 1px solid var(--border); }

footer { text-align: center; margin-top: 3rem; padding: 1rem;
         color: var(--text-dim); font-size: 0.85rem;
         border-top: 1px solid var(--border); }
"""


# =====================================================================
# PromptFitter-specific report
# =====================================================================


def generate_fit_report(
    result: Any,  # PromptFitResult
    output_path: str | Path | None = None,
) -> str:
    """Generate a self-contained HTML report for ``PromptFitResult``.

    Includes KPI cards, parameter change table, failure clusters,
    dimension comparison, trial history, prompt diff, few-shot examples,
    and actionable recommendations.

    Args:
        result: ``PromptFitResult`` from ``PromptFitter.fit()``.
        output_path: Path to write the HTML file. Auto-generated if None.

    Returns:
        Path to the generated report file.
    """
    if output_path is None:
        output_path = Path(
            f".optimize/{result.agent}/fit_report_{result.experiment_id}.html"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build HTML
    html_content = _build_fit_report_html(result)
    output_path.write_text(html_content, encoding="utf-8")
    logger.info(f"📊 Fit report generated: {output_path}")
    return str(output_path)


def _build_fit_report_html(result: Any) -> str:
    """Build the self-contained HTML for a PromptFitResult."""
    improvement = result.best_score - result.baseline_score
    improvement_pct = (
        (improvement / result.baseline_score * 100) if result.baseline_score > 0 else 0
    )

    # KPI section
    kpis = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 2rem 0;">
      <div class="kpi-card">
        <div style="color: #8b949e; font-size: 0.85rem;">Baseline Score</div>
        <div style="font-size: 2rem; font-weight: 700; color: #f0883e;">{result.baseline_score:.4f}</div>
      </div>
      <div class="kpi-card">
        <div style="color: #8b949e; font-size: 0.85rem;">Best Score</div>
        <div style="font-size: 2rem; font-weight: 700; color: #3fb950;">{result.best_score:.4f}</div>
      </div>
      <div class="kpi-card">
        <div style="color: #8b949e; font-size: 0.85rem;">Absolute Improvement</div>
        <div style="font-size: 2rem; font-weight: 700; color: {'#3fb950' if improvement > 0 else '#f85149'};">{improvement:+.4f}</div>
      </div>
      <div class="kpi-card">
        <div style="color: #8b949e; font-size: 0.85rem;">Relative Improvement</div>
        <div style="font-size: 2rem; font-weight: 700; color: {'#3fb950' if improvement_pct > 0 else '#f85149'};">{improvement_pct:+.1f}%</div>
      </div>
      <div class="kpi-card">
        <div style="color: #8b949e; font-size: 0.85rem;">Total Trials</div>
        <div style="font-size: 2rem; font-weight: 700; color: #58a6ff;">{len(result.trials)}</div>
      </div>
      <div class="kpi-card">
        <div style="color: #8b949e; font-size: 0.85rem;">Duration</div>
        <div style="font-size: 2rem; font-weight: 700; color: #d2a8ff;">{result.duration_seconds:.1f}s</div>
      </div>
    </div>
    """

    # Param suggestions table
    param_rows = ""
    for name, delta in result.param_suggestions.items():
        param_rows += f"""
        <tr>
          <td><code>{html.escape(str(name))}</code></td>
          <td style="color: #f0883e;">{html.escape(str(delta.old_value))}</td>
          <td style="color: #3fb950;">{html.escape(str(delta.new_value))}</td>
          <td style="color: #8b949e;">{html.escape(str(delta.reason))}</td>
        </tr>"""

    param_section = ""
    if result.param_suggestions:
        param_section = f"""
        <h2>🔧 Parameter Changes</h2>
        <table>
          <thead><tr><th>Parameter</th><th>Old</th><th>New</th><th>Reason</th></tr></thead>
          <tbody>{param_rows}</tbody>
        </table>
        """

    # Metric deltas table
    delta_rows = ""
    for dim, delta_val in result.metric_deltas.items():
        color = "#3fb950" if delta_val >= 0 else "#f85149"
        delta_rows += f"""
        <tr>
          <td>{html.escape(dim)}</td>
          <td style="color: {color};">{delta_val:+.4f}</td>
        </tr>"""

    delta_section = ""
    if result.metric_deltas:
        delta_section = f"""
        <h2>📊 Metric Deltas</h2>
        <table>
          <thead><tr><th>Dimension</th><th>Delta</th></tr></thead>
          <tbody>{delta_rows}</tbody>
        </table>
        """

    # Failure clusters
    cluster_items = ""
    for cluster in result.failure_clusters:
        if isinstance(cluster, dict):
            label = cluster.get("label", "unknown")
            desc = cluster.get("description", "")
            count = cluster.get("count", 0)
            fix = cluster.get("suggested_fix", "")
        else:
            label = getattr(cluster, "label", "unknown")
            desc = getattr(cluster, "description", "")
            count = getattr(cluster, "count", 0)
            fix = getattr(cluster, "suggested_fix", "")
        cluster_items += f"""
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
          <strong style="color: #f0883e;">{html.escape(str(label))}</strong>
          <span style="color: #8b949e;"> ({count} cases)</span>
          <p style="color: #c9d1d9; margin: 0.5rem 0;">{html.escape(str(desc))}</p>
          <p style="color: #3fb950; font-style: italic;">💡 {html.escape(str(fix))}</p>"""

        # Show affected params if available
        affected = (
            cluster.get("affected_params", []) if isinstance(cluster, dict)
            else getattr(cluster, "affected_params", [])
        )
        gains = (
            cluster.get("expected_metric_gain", {}) if isinstance(cluster, dict)
            else getattr(cluster, "expected_metric_gain", {})
        )
        if affected:
            params_html = ", ".join(f"<code>{html.escape(p)}</code>" for p in affected)
            cluster_items += f"""
          <p style="color: #d2a8ff; margin: 0.25rem 0; font-size: 0.9em;">🎯 Affected params: {params_html}</p>"""
        if gains:
            gains_parts = [f"{k}: <strong>{v:+.2f}</strong>" for k, v in gains.items()]
            cluster_items += f"""
          <p style="color: #58a6ff; margin: 0.25rem 0; font-size: 0.9em;">📈 Expected gain: {', '.join(gains_parts)}</p>"""

        cluster_items += """
        </div>"""

    cluster_section = ""
    if result.failure_clusters:
        cluster_section = f"""
        <h2>🔍 Failure Clusters</h2>
        {cluster_items}
        """

    # Suggestions
    suggestion_items = "".join(
        f"<li>{html.escape(s)}</li>" for s in result.suggestions
    )
    suggestion_section = ""
    if result.suggestions:
        suggestion_section = f"""
        <h2>💡 Recommendations</h2>
        <ol style="color: #c9d1d9; line-height: 1.8;">{suggestion_items}</ol>
        """

    # Prompt diff
    baseline_prompt = result.baseline_config.system_prompt
    best_prompt = result.best_config.system_prompt
    diff_lines = list(
        difflib.unified_diff(
            baseline_prompt.splitlines(keepends=True),
            best_prompt.splitlines(keepends=True),
            fromfile="baseline",
            tofile="best",
            lineterm="",
        )
    )
    diff_html = ""
    for line in diff_lines:
        escaped = html.escape(line.rstrip("\n"))
        if line.startswith("+") and not line.startswith("+++"):
            diff_html += f'<div style="color: #3fb950; background: rgba(63,185,80,0.1);">  {escaped}</div>'
        elif line.startswith("-") and not line.startswith("---"):
            diff_html += f'<div style="color: #f85149; background: rgba(248,81,73,0.1);">  {escaped}</div>'
        else:
            diff_html += f"<div>  {escaped}</div>"

    diff_section = f"""
    <h2>📝 Prompt Diff</h2>
    <div style="background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 1rem; font-family: monospace; font-size: 0.85rem; overflow-x: auto; white-space: pre-wrap;">
{diff_html}
    </div>
    """

    # Few-shot examples
    few_shot_section = ""
    if result.best_config.few_shot_examples:
        examples_html = ""
        for i, ex in enumerate(result.best_config.few_shot_examples, 1):
            q = html.escape(str(ex.get("query", "")))
            r = html.escape(str(ex.get("response", "")))
            examples_html += f"""
            <div style="background: #161b22; border-left: 3px solid #58a6ff; padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;">
              <div style="color: #58a6ff; font-weight: 600;">Example {i}</div>
              <div style="color: #8b949e;">Q: {q}</div>
              <div style="color: #c9d1d9;">A: {r[:300]}{'...' if len(r) > 300 else ''}</div>
            </div>"""
        few_shot_section = f"""
        <h2>📚 Few-Shot Examples</h2>
        {examples_html}
        """

    # Deployment recommendation section
    deployment_section = ""
    if getattr(result, "deployment_recommendation", None):
        rec = result.deployment_recommendation
        safety_html = ""
        if rec.safety_notes:
            safety_items = "".join(
                f"<li style='color: #f0883e;'>{html.escape(n)}</li>"
                for n in rec.safety_notes
            )
            safety_html = f"<ul>{safety_items}</ul>"

        deployment_section = f"""
        <h2>🚀 Deployment Recommendation</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1rem 0;">
          <div class="kpi-card">
            <div style="color: #8b949e; font-size: 0.85rem;">Version</div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #58a6ff;">{html.escape(rec.prompt_version)}</div>
          </div>
          <div class="kpi-card">
            <div style="color: #8b949e; font-size: 0.85rem;">Confidence</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {'#3fb950' if rec.confidence == 'high' else '#f0883e' if rec.confidence == 'medium' else '#f85149'};">{rec.confidence}</div>
          </div>
          <div class="kpi-card">
            <div style="color: #8b949e; font-size: 0.85rem;">Rollout Strategy</div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #d2a8ff;">{rec.rollout.strategy} @ {rec.rollout.initial_weight:.0%}</div>
          </div>
          <div class="kpi-card">
            <div style="color: #8b949e; font-size: 0.85rem;">Monitor</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #c9d1d9;">{rec.rollout.monitoring_hours}h</div>
          </div>
        </div>
        {safety_html}
        """

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PromptFitter Report — {html.escape(result.agent)}</title>
  <style>
    :root {{ --bg: #0d1117; --surface: #161b22; --border: #30363d; --text: #c9d1d9; --text-dim: #8b949e; }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; padding: 2rem; max-width: 1200px; margin: 0 auto; line-height: 1.6; }}
    h1 {{ font-size: 1.8rem; color: #f0f6fc; margin-bottom: 0.5rem; }}
    h2 {{ font-size: 1.3rem; color: #f0f6fc; margin: 2rem 0 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
    .kpi-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; text-align: center; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ background: #1c2128; color: var(--text-dim); font-weight: 600; font-size: 0.85rem; text-transform: uppercase; }}
    td {{ color: var(--text); }}
    code {{ background: #1c2128; padding: 0.2em 0.4em; border-radius: 4px; font-size: 0.9em; }}
    footer {{ text-align: center; margin-top: 3rem; padding: 1rem; color: var(--text-dim); font-size: 0.85rem; border-top: 1px solid var(--border); }}
  </style>
</head>
<body>
  <h1>⚡ PromptFitter Report</h1>
  <p style="color: #8b949e;">Agent: <strong style="color: #58a6ff;">{html.escape(result.agent)}</strong> | Experiment: <code>{html.escape(result.experiment_id)}</code> | {timestamp}</p>

  {kpis}
  {param_section}
  {delta_section}
  {cluster_section}
  {suggestion_section}
  {deployment_section}
  {diff_section}
  {few_shot_section}

  <footer>Generated by Agentomatic PromptFitter — {timestamp}</footer>
</body>
</html>"""

