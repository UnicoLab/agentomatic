#!/usr/bin/env python3
"""Generate the Keras-style loss-curve SVG from real showcase summaries.

Reads the ``summary_<mode>.json`` files produced by ``train.py`` and renders
a self-contained SVG line chart (loss per epoch, one line per optimizer
mode) — no matplotlib needed. Used by the docs.

Usage::

    python examples/keras_optimize_showcase/make_chart.py docs/assets/showcase/summary_rewrite.json ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

W, H, PAD_L, PAD_R, PAD_T, PAD_B = 980, 460, 60, 30, 30, 60
COLORS = {
    "rewrite": "#58a6ff",
    "gepa_like": "#f0883e",
    "mipro_like": "#3fb950",
    "few_shot_bootstrap": "#d2a8ff",
    "param_search": "#f85149",
    "default": "#8b949e",
}


def load_curve(path: Path) -> tuple[str, list[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mode = str(data.get("mode", path.stem.replace("summary_", "")))
    loss = [float(v) for v in data["history"]["history"].get("loss", [])]
    return mode, loss


def render(curves: list[tuple[str, list[float]]], out: Path) -> None:
    all_vals = [v for _, vals in curves for v in vals]
    y_max = max(all_vals, default=1.0)
    y_min = min(0.0, min(all_vals, default=0.0))
    span = max(y_max - y_min, 1e-9)
    n_points = max(len(vals) for _, vals in curves)
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B

    def x_for(i: int, n: int) -> float:
        return PAD_L + (plot_w * i / max(n - 1, 1))

    def y_for(v: float) -> float:
        return PAD_T + plot_h * (1 - (v - y_min) / span)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">'
    )
    parts.append(
        f'<rect width="{W}" height="{H}" fill="#0d1117" rx="12"/>'
    )

    # gridlines + y labels
    for g in range(6):
        frac = g / 5
        gy = PAD_T + plot_h * frac
        val = y_max - frac * span
        parts.append(
            f'<line x1="{PAD_L}" y1="{gy:.1f}" x2="{W - PAD_R}" y2="{gy:.1f}" '
            f'stroke="#21262d" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 10}" y="{gy + 4:.1f}" text-anchor="end" '
            f'font-size="12" fill="#8b949e">{val:.2f}</text>'
        )

    # x labels (epochs)
    n = n_points
    for i in range(n):
        label = "base" if i == 0 else str(i)
        parts.append(
            f'<text x="{x_for(i, n):.1f}" y="{H - PAD_B + 20}" text-anchor="middle" '
            f'font-size="12" fill="#8b949e">{label}</text>'
        )

    # lines
    for mode, vals in curves:
        color = COLORS.get(mode, COLORS["default"])
        pts = " ".join(
            f"{x_for(i, n):.1f},{y_for(v):.1f}" for i, v in enumerate(vals)
        )
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        # point markers
        for i, v in enumerate(vals):
            parts.append(
                f'<circle cx="{x_for(i, n):.1f}" cy="{y_for(v):.1f}" r="3.5" fill="{color}"/>'
            )
        # final value label
        last = vals[-1]
        parts.append(
            f'<text x="{x_for(n - 1, n) + 8:.1f}" y="{y_for(last) + 4:.1f}" '
            f'font-size="12" fill="{color}">{mode} {last:.3f}</text>'
        )

    # legend
    lx = PAD_L
    for mode, _ in curves:
        color = COLORS.get(mode, COLORS["default"])
        parts.append(
            f'<rect x="{lx}" y="{H - PAD_B + 32}" width="12" height="12" rx="2" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{lx + 18}" y="{H - PAD_B + 43}" font-size="13" fill="#c9d1d9">{mode}</text>'
        )
        lx += 26 + 12 * len(mode) + 18

    parts.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out}")


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if not args:
        print(__doc__)
        return 1
    summaries = [Path(a) for a in args if a.endswith(".json")]
    curves = [load_curve(p) for p in summaries]
    out = Path("docs/assets/showcase/loss_curves.svg")
    render(curves, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
