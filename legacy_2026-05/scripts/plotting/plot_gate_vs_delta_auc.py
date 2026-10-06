#!/usr/bin/env python3
"""
plot_gate_vs_delta_auc: Fig. 3 gate value vs CNN marginal gain
================================================================
Task-level aligned profile:
  left panel  = U1_Unified gate_mean, sorted by task-level mean
  right panel = CNN marginal gain, M4_CNN_Full − M2_MLP_360

xG is included as a reference row with ΔR² in the right panel.
Pearson r uses only the 8 binary tasks.

Inputs:
  data/results_all.json

Outputs:
  figures/fig3.png / .pdf

Usage:
  PYTHONPATH=. python scripts/plotting/plot_gate_vs_delta_auc.py

"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.plotting.plot_utils import set_paper_style

CACHE = _PROJECT_ROOT / "data"
FIG = _PROJECT_ROOT / "figures"; FIG.mkdir(exist_ok=True)

BINARY_TASKS = ["pass", "dribble", "ball_receipt", "shot", "duel",
                "interception", "ball_recovery", "pressure"]
TASK_LABELS = {
    "pass": "Pass",
    "dribble": "Dribble",
    "ball_receipt": "Ball Receipt",
    "shot": "Shot",
    "duel": "Duel",
    "interception": "Interception",
    "ball_recovery": "Ball Recovery",
    "pressure": "Pressure",
    "xg": "xG (reg.)",
}

TASK_FULL_LABELS = {
    "pass": "Pass Success",
    "dribble": "Dribble Success",
    "ball_receipt": "Ball Receipt",
    "shot": "Shot Goal",
    "duel": "Duel Won",
    "interception": "Interception",
    "ball_recovery": "Ball Recovery",
    "pressure": "Pressure Turnover",
    "xg": "xG (regression)",
}


def collect_metrics(rows, task, model, is_xg=False):
    for r in rows:
        if r["task"] == task and r["model"] == model:
            if is_xg:
                return r.get("test_r2", None), r.get("gate_mean", None)
            return r.get("test_auc", None), r.get("gate_mean", None)
    return None, None


def main():
    set_paper_style()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.8,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.8,
        "axes.linewidth": 1.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })
    rows = json.loads((CACHE / "results_all.json").read_text())

    records = []
    for t in BINARY_TASKS:
        m4, _ = collect_metrics(rows, t, "M4_CNN_Full")
        m2, _ = collect_metrics(rows, t, "M2_MLP_360")
        _, g_mean = collect_metrics(rows, t, "U1_Unified")
        if m4 is None or m2 is None or g_mean is None:
            continue
        records.append({
            "task": t,
            "label": TASK_FULL_LABELS[t],
            "gate": g_mean,
            "gain": m4 - m2,
            "is_xg": False,
        })

    # xG separately (regression, ΔR²)
    r4, _ = collect_metrics(rows, "xg", "M4_CNN_Full", is_xg=True)
    r2_, _ = collect_metrics(rows, "xg", "M2_MLP_360", is_xg=True)
    _, g_xg = collect_metrics(rows, "xg", "U1_Unified")
    xg_dx = (r4 - r2_) if (r4 is not None and r2_ is not None) else None
    if xg_dx is not None and g_xg is not None:
        records.append({
            "task": "xg",
            "label": TASK_FULL_LABELS["xg"],
            "gate": g_xg,
            "gain": xg_dx,
            "is_xg": True,
        })

    records = sorted(records, key=lambda r_: r_["gate"], reverse=True)
    binary_records = [r_ for r_ in records if not r_["is_xg"]]
    xs = np.array([r_["gain"] for r_ in binary_records])
    ys = np.array([r_["gate"] for r_ in binary_records])

    # Pearson on binary tasks only
    if len(xs) >= 2:
        pearson = pearsonr(xs, ys)
        r = float(pearson.statistic)
        p = float(pearson.pvalue)
    else:
        r = float("nan")
        p = float("nan")

    if len(xs) >= 2:
        slope, intercept = np.polyfit(xs, ys, 1)
    else:
        slope, intercept = float("nan"), float("nan")

    NORMAL_COLOR = "#1F4E79"
    DUEL_COLOR = "#CC3311"
    XG_COLOR = "#EE7733"
    POS_COLOR = "#1F4E79"
    NEG_COLOR = "#B8B8B8"
    ZERO_COLOR = "#5F5F5F"
    GRID_COLOR = "#E6E6E6"

    fig, (ax_gate, ax_gain) = plt.subplots(
        ncols=2,
        figsize=(8.2, 4.9),
        dpi=300,
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.02, 1.25], "wspace": 0.10},
    )

    y_pos = np.arange(len(records))[::-1]
    y_lookup = {r_["task"]: y for r_, y in zip(records, y_pos)}

    for ax in (ax_gate, ax_gain):
        for y in y_pos:
            ax.axhline(y, color=GRID_COLOR, linewidth=0.7, zorder=0)
        ax.tick_params(axis="y", length=0)

    for rec in records:
        y = y_lookup[rec["task"]]
        marker = "^" if rec["is_xg"] else "o"
        color = XG_COLOR if rec["is_xg"] else (
            DUEL_COLOR if rec["task"] == "duel" else NORMAL_COLOR
        )
        ax_gate.plot([0.20, rec["gate"]], [y, y],
                     color="#D7DEE8", linewidth=2.1, solid_capstyle="round",
                     zorder=1)
        ax_gate.scatter(rec["gate"], y, s=88, marker=marker, facecolor=color,
                        edgecolor="white", linewidth=0.9, zorder=3)
        ax_gate.scatter(rec["gate"], y, s=108, marker=marker, facecolor="none",
                        edgecolor="#222222", linewidth=0.65, zorder=3)

        gain_color = XG_COLOR if rec["is_xg"] else (
            DUEL_COLOR if rec["task"] == "duel" else
            (POS_COLOR if rec["gain"] >= 0 else NEG_COLOR)
        )
        ax_gain.plot([0.0, rec["gain"]], [y, y],
                     color=gain_color, linewidth=2.1, solid_capstyle="round",
                     alpha=0.82, zorder=1)
        ax_gain.scatter(rec["gain"], y, s=88, marker=marker,
                        facecolor=gain_color, edgecolor="white",
                        linewidth=0.9, zorder=3)
        ax_gain.scatter(rec["gain"], y, s=108, marker=marker,
                        facecolor="none", edgecolor="#222222",
                        linewidth=0.65, zorder=3)

    ax_gain.axvline(0.0, color=ZERO_COLOR, linestyle="--",
                    linewidth=1.0, zorder=0)

    ax_gate.set_yticks(y_pos)
    ax_gate.set_yticklabels([r_["label"] for r_ in records])
    ax_gain.tick_params(axis="y", labelleft=False)

    ax_gate.set_xlim(0.20, 0.47)
    ax_gate.set_xticks([0.20, 0.30, 0.40])
    ax_gate.set_xlabel("U1 gate mean")
    ax_gate.set_title("(a) Learned CNN-branch allocation", loc="left",
                      fontweight="bold", pad=8)

    ax_gain.set_xlim(-0.07, 0.115)
    ax_gain.set_xticks([-0.05, 0.00, 0.05, 0.10])
    ax_gain.set_xlabel("CNN marginal gain (M4 - M2)")
    ax_gain.set_title("(b) Marginal gain from CNN input", loc="left",
                      fontweight="bold", pad=8)

    png = FIG / "fig3.png"
    pdf = FIG / "fig3.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {png} and {pdf}")
    print(f"  N binary tasks = {len(xs)}, Pearson r = {r:.4f}, p = {p:.4f}")
    print(f"  Slope = {slope:.4f}, intercept = {intercept:.4f}")
    for rec in records:
        metric = "ΔR²" if rec["is_xg"] else "ΔAUC"
        print(f"    {TASK_LABELS[rec['task']]:14s}  "
              f"{metric}={rec['gain']:+.4f}  gate={rec['gate']:.3f}")


if __name__ == "__main__":
    main()
