#!/usr/bin/env python3
"""
plot_strict_vs_relaxed: Fig 7 NEW: strict vs relaxed holdout bar chart
========================================================================
Section 6.5, Table 13 -> visualise the covariate-shift remediation effect.

4 rows (combos) x 3 columns (AUC, Brier, ECE) grouped bar:
  combos = {Dribble M4, Dribble G1, Duel M4, Duel G1}
  bars   = strict (grey) vs relaxed (blue/green)
  ECE column adds a horizontal line at 0.10 (severe-miscalibration threshold)

All numbers are computed on the fly from _test_calib.npz (strict, probs_before)
versus _relaxed.npz (relaxed, probs), ensuring they share the same source as
the Fig 4 reliability diagrams (relaxed npz / strict probs_before).

Inputs:
  data/predictions/{task}_{model}_test_calib.npz
  data/predictions/{task}_{model}_relaxed.npz

Output:
  figures/figS4.png / .pdf

Usage:
  PYTHONPATH=. python scripts/supplementary_plots/plot_strict_vs_relaxed.py

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from scripts.plotting.plot_utils import set_paper_style

CACHE = Path("data/predictions")
FIG = Path("figures"); FIG.mkdir(exist_ok=True)

COMBOS = [
    ("dribble", "M4_CNN_2ch", "Dribble × M4 CNN"),
    ("dribble", "G1_Gating_2ch", "Dribble × G1 Gating"),
    ("duel",    "M4_CNN_2ch", "Duel × M4 CNN"),
    ("duel",    "G1_Gating_2ch", "Duel × G1 Gating"),
]


def ece(y_true, y_prob, n_bins=15):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    n = len(y_prob)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        cnt = int(m.sum())
        if cnt:
            e += (cnt / n) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(e)


def metrics_strict(task, model):
    d = np.load(CACHE / f"{task}_{model}_test_calib.npz", allow_pickle=True)
    y = d["y_true"].astype(float); p = d["probs_before"].astype(float)
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "n": len(y),
    }


def metrics_relaxed(task, model):
    d = np.load(CACHE / f"{task}_{model}_relaxed.npz", allow_pickle=True)
    y = d["y_test"].astype(float); p = d["probs"].astype(float)
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "n": len(y),
    }


def main():
    set_paper_style()
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    rows = []
    for task, model, _ in COMBOS:
        s = metrics_strict(task, model)
        r = metrics_relaxed(task, model)
        rows.append({"task": task, "model": model, "strict": s, "relaxed": r})
        print(f"{task:>10s} {model:>16s} strict[n={s['n']}] auc={s['auc']:.3f} "
              f"brier={s['brier']:.3f} ece={s['ece']:.3f}  | "
              f"relaxed[n={r['n']}] auc={r['auc']:.3f} brier={r['brier']:.3f} "
              f"ece={r['ece']:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), dpi=150, sharey=False)
    metric_keys = ["auc", "brier", "ece"]
    metric_titles = ["AUC (higher = better)",
                     "Brier (lower = better)",
                     "ECE (lower = better)"]
    color_strict = "#888888"   # gray: matches main Fig 5 COLOR_STRICT
    color_relaxed = "#1F4E79"  # Tol deep blue: matches main Fig 5 COLOR_RELAXED

    n_combos = len(COMBOS)
    bar_w = 0.36
    x = np.arange(n_combos)

    for k, (mkey, mtitle) in enumerate(zip(metric_keys, metric_titles)):
        ax = axes[k]
        s_vals = [r["strict"][mkey] for r in rows]
        r_vals = [r["relaxed"][mkey] for r in rows]
        b1 = ax.bar(x - bar_w / 2, s_vals, bar_w, color=color_strict,
                    edgecolor="#444", linewidth=0.6, label="strict (24/25 only)")
        b2 = ax.bar(x + bar_w / 2, r_vals, bar_w, color=color_relaxed,
                    edgecolor="#0F2C4A", linewidth=0.6, label="relaxed (+80% 24/25 in train)")

        for bars, vals in [(b1, s_vals), (b2, r_vals)]:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + max(s_vals + r_vals) * 0.012,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels([c[2] for c in COMBOS], rotation=15, ha="right",
                           fontsize=9.5)
        ax.set_title(mtitle, fontsize=11.5, loc="left")
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.set_axisbelow(True)
        ymax = max(s_vals + r_vals) * 1.18
        ax.set_ylim(0, ymax)

        if mkey == "ece":
            ax.axhline(0.10, color="#CC3311", linestyle="--", linewidth=1.0,
                        label="severe miscalibration (0.10)", zorder=5)

        if k == 0:
            ax.set_ylabel("Score", fontsize=10.5)

    handles, labels = axes[0].get_legend_handles_labels()
    handles_ece, labels_ece = axes[2].get_legend_handles_labels()
    # Append ECE-only items to legend if present
    for h, l in zip(handles_ece, labels_ece):
        if l not in labels:
            handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), fontsize=10, frameon=False)

    fig.suptitle("Strict vs relaxed holdout: covariate-shift remediation effect (4 combos × 3 metrics)",
                 fontsize=13, y=1.07)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    png = FIG / "figS4.png"
    pdf = FIG / "figS4.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {png} and {pdf}")


if __name__ == "__main__":
    main()
