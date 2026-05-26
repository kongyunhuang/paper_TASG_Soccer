#!/usr/bin/env python3
"""
plot_reliability_diagrams: reliability diagrams (Dribble/Duel x M4/G1 x strict/relaxed)
=========================================================================================
Paper figure: visualises the 2026-04-17 covariate-shift finding:
  strict holdout (22/23+23/24 -> 24/25): severely below the diagonal (the model over-predicts)
  relaxed holdout (+80% of 24/25):       almost on the diagonal

Inputs:
  data/predictions/{task}_{model}_test_calib.npz  (strict, from fit_temperature)
  data/predictions/{task}_{model}_relaxed.npz     (relaxed)

Output:
  figures/figS3.png / .pdf  (4x2 grid)

Usage:
  PYTHONPATH=. python scripts/supplementary_plots/plot_reliability_diagrams.py

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.plotting.plot_utils import set_paper_style

CACHE = Path("data/predictions")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

COMBOS = [
    ("dribble", "M4_CNN_2ch"),
    ("dribble", "G1_Gating_2ch"),
    ("duel", "M4_CNN_2ch"),
    ("duel", "G1_Gating_2ch"),
]


def reliability_bins(y_true, y_prob, n_bins=12):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_conf = np.full(n_bins, np.nan)
    mean_acc = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            m = (y_prob >= lo) & (y_prob <= hi)
        else:
            m = (y_prob >= lo) & (y_prob < hi)
        counts[i] = int(m.sum())
        if counts[i]:
            mean_conf[i] = y_prob[m].mean()
            mean_acc[i] = y_true[m].mean()
    return centers, mean_conf, mean_acc, counts


def ece(y_true, y_prob, n_bins=15):
    _, mc, ma, cnt = reliability_bins(y_true, y_prob, n_bins)
    valid = cnt > 0
    return float((cnt[valid] / cnt.sum() * np.abs(ma[valid] - mc[valid])).sum())


def load_strict(task, model):
    d = np.load(CACHE / f"{task}_{model}_test_calib.npz", allow_pickle=True)
    return d["y_true"].astype(float), d["probs_before"].astype(float)


def load_relaxed(task, model):
    p = CACHE / f"{task}_{model}_relaxed.npz"
    if not p.exists():
        return None, None
    d = np.load(p, allow_pickle=True)
    return d["y_test"].astype(float), d["probs"].astype(float)


def draw_panel(ax, y, p, title):
    if y is None:
        ax.text(0.5, 0.5, "predictions not\navailable yet",
                ha="center", va="center", fontsize=10, color="#888")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        return
    centers, mc, ma, cnt = reliability_bins(y, p, n_bins=12)
    valid = cnt > 0
    ax.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=0.8, label="perfect")
    # Light-grey bars for bin counts (more transparent: 0.6 -> 0.4)
    pop = cnt / max(cnt.max(), 1) * 0.25
    ax.bar(centers, pop, width=1 / len(centers) * 0.85,
           color="#cccccc", edgecolor="none", alpha=0.4, zorder=0)
    ax.plot(mc[valid], ma[valid], "o-", color="#CC3311", linewidth=1.8,
            markersize=6, label="observed", zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    e = ece(y, p, n_bins=15)
    mean_y = y.mean(); mean_p = p.mean()
    ax.text(0.03, 0.95,
            f"ECE = {e:.3f}\n$\\bar{{y}}$ = {mean_y:.2f},  $\\bar{{p}}$ = {mean_p:.2f}",
            fontsize=10, va="top",
            bbox=dict(facecolor="white", edgecolor="#aaa", boxstyle="round,pad=0.35"))
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, linestyle=":")


def main():
    set_paper_style()
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    # 2 rows (strict / relaxed) × 4 cols (combos): 14 × 8.2
    fig, axes = plt.subplots(2, len(COMBOS), figsize=(15, 8.2),
                             dpi=150, sharex=True, sharey=True)

    row_titles = ["strict holdout  (train 22/23+23/24, test 24/25)",
                  "relaxed holdout  (+80% of 24/25 added to train)"]

    for col, (task, model) in enumerate(COMBOS):
        y_s, p_s = load_strict(task, model)
        y_r, p_r = load_relaxed(task, model)
        col_title = f"{task} × {model.replace('_2ch','')}"
        draw_panel(axes[0, col], y_s, p_s, col_title)
        draw_panel(axes[1, col], y_r, p_r, col_title)

    # Row labels (LHS)
    for r, label in enumerate(row_titles):
        axes[r, 0].set_ylabel(f"{label}\n\nempirical positive rate",
                               fontsize=10.5)
    for ax in axes[1, :]:
        ax.set_xlabel("predicted probability", fontsize=10.5)

    fig.suptitle("Reliability diagrams: covariate-shift diagnostic\n"
                 "(rows = strict vs relaxed holdout; light-grey bars = bin populations)",
                 fontsize=13, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_png = FIG_DIR / "figS3.png"
    out_pdf = FIG_DIR / "figS3.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
