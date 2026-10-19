#!/usr/bin/env python
"""Draw the architecture diagram used in the report -> reports/figures/architecture.png"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from heartrisk.plotting import AQUA, BLUE, INK, INK_SECONDARY, ORANGE, SURFACE, apply_style  # noqa: E402

W, H = 2.1, 0.95        # box size
STEP = 2.45             # horizontal distance between box origins
X0 = 2.1                # x of first box
ROWS = {"data": 8.3, "ci": 5.7, "serve": 3.2, "mon": 0.8}
TINT = {BLUE: "#e8f0fb", ORANGE: "#fdeee8", AQUA: "#e5f6ef", INK_SECONDARY: "#efeeeb"}


def box(ax, col, row, text, color):
    x, y = X0 + col * STEP, ROWS[row]
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=1.6, edgecolor=color, facecolor=TINT[color]))
    ax.text(x + W / 2, y + H / 2, text, ha="center", va="center", fontsize=8.6, color=INK, linespacing=1.35)
    return x, y


def arrow(ax, p, q, rad=0.0, style="-|>", ls="-", color=INK_SECONDARY):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, linewidth=1.4,
                                 color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


def elbow(ax, start, end, mid_y, ls="--", color=INK_SECONDARY):
    """Right-angle connector: down from start, across at mid_y, down into end."""
    (x1, y1), (x2, y2) = start, end
    ax.plot([x1, x1, x2], [y1, mid_y, mid_y], color=color, linewidth=1.4, linestyle=ls, solid_capstyle="round")
    arrow(ax, (x2, mid_y), (x2, y2), ls=ls, color=color)


def chain(ax, row, labels, color, col0=0):
    pos = [box(ax, col0 + i, row, t, color) for i, t in enumerate(labels)]
    for (x1, y1), (x2, y2) in zip(pos, pos[1:]):
        arrow(ax, (x1 + W, y1 + H / 2), (x2, y2 + H / 2))
    return pos


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(17, 9.8))
    ax.set_xlim(0, 17)
    ax.set_ylim(0.1, 10.2)
    ax.axis("off")
    ax.grid(False)

    lanes = [("data", "1  DATA &\nEXPERIMENTS", BLUE), ("ci", "2  CI / CD", ORANGE),
             ("serve", "3  SERVING\n(Kubernetes)", AQUA), ("mon", "4  MONITORING", INK_SECONDARY)]
    for row, label, color in lanes:
        y = ROWS[row]
        ax.add_patch(FancyBboxPatch((0.15, y - 0.35), 16.6, H + 0.7, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    linewidth=0, facecolor="#f6f5f2", zorder=0))
        ax.text(0.35, y + H / 2, label, ha="left", va="center", fontsize=10, fontweight="bold", color=color)

    data = chain(ax, "data", [
        "UCI Heart Disease\nCleveland, 303 rows", "prepare_data.py\nclean + binarise target",
        "EDA\n4 figures + summary", "train.py\nnested CV + grid search\n3 models",
        "Saved pipeline\njoblib + model card"], BLUE)
    mx, my = X0 + 3 * STEP, 7.0
    ax.add_patch(FancyBboxPatch((mx, my), W, 0.72, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=1.6, edgecolor=BLUE, facecolor=TINT[BLUE]))
    ax.text(mx + W / 2, my + 0.36, "MLflow: runs, plots,\nregistry (SQLite)", ha="center", va="center", fontsize=8.6)
    arrow(ax, (data[3][0] + W / 2, data[3][1]), (mx + W / 2, my + 0.72), style="<|-")
    ax.text(mx + W + 0.15, my + 0.36, "params, metrics,\nartifacts, models", fontsize=7.8,
            color=INK_SECONDARY, va="center")

    ci = chain(ax, "ci", [
        "git push / PR\nGitHub", "flake8\nlint", "pytest\n43 tests",
        "train + MLflow\n(retrains in CI)", "quality gate\nROC-AUC >= 0.80",
        "docker build\n+ smoke test"], ORANGE)
    ax.text(X0, ROWS["ci"] - 0.55, "Every step fails the pipeline on error; logs, models, reports and mlflow.db are "
            "uploaded as run artifacts.", fontsize=8.2, color=INK_SECONDARY)

    serve = chain(ax, "serve", [
        "Docker image\nFastAPI + model baked in", "Deployment\n2 pods, probes, limits, HPA",
        "Service LoadBalancer\n+ Ingress", "Client\nPOST /predict (JSON)"], AQUA)
    elbow(ax, (ci[5][0] + W / 2, ci[5][1]), (serve[0][0] + W / 2, serve[0][1] + H), mid_y=4.6)
    ax.text(X0 + 2.9 * STEP, 4.68, "image is built from the CI-trained model", fontsize=7.8, color=INK_SECONDARY)
    box(ax, 4, "serve", "Response\nprediction + confidence", AQUA)
    arrow(ax, (serve[3][0] + W, serve[3][1] + H / 2), (X0 + 4 * STEP, serve[3][1] + H / 2))

    mon = chain(ax, "mon", [
        "FastAPI pods\nJSON logs + /metrics", "Prometheus\nscrapes every 5 s",
        "Grafana dashboard\nrate, latency, errors,\ndrift signal"], INK_SECONDARY, col0=1)
    arrow(ax, (serve[1][0] + W / 2, serve[1][1]), (mon[0][0] + W / 2, mon[0][1] + H), ls="--")

    ax.text(0.15, 10.0, "HeartRisk: end-to-end MLOps architecture", fontsize=14, fontweight="bold", color=INK)
    fig.patch.set_facecolor(SURFACE)
    out = ROOT / "reports" / "figures" / "architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
