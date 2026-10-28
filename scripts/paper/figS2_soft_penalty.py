#!/usr/bin/env python3
"""
figS2_soft_penalty.py, soft gate with an opening penalty, selection step and confirmation step
==============================================================================================
Supplementary figure. The selection step is the path of seed 0 along λ, the confirmation step is λ = 0.01 with seeds 1,
2 and 3.
  Left panel   mean gate of the outcome tasks and destination gate, for enriched and for thin scalars, one line of seed 0
               each; the three seeds of the confirmation step are drawn as points with their mean marked.
  Right panel  mean loss of the outcome tasks after shuffling the whole tensor and mean loss with the gate forced closed
               at test time; in the selection step with thin scalars only λ = 0.01 has these two ablations.
All numbers come from the section on the gate penalty experiments in results/u1_clean_summary.md (the four tables of the
selection and confirmation steps of the soft gate with enriched and with thin scalars), registered by line number.
The dotted lines are the two thresholds of the rule fixed in advance for the soft gate with enriched scalars (outcome gate
mean at most 0.10, loss under a whole-tensor shuffle at most 0.01).

Inputs   results/u1_clean_summary.md
Outputs  outputs/figures/figS2_soft_penalty.{pdf,png}, results/figure_data/figS2_soft_penalty_{data,labels}.csv,
         outputs/provenance/figS2_soft_penalty_sources.json

Usage
  PYTHONPATH=. python -u scripts/paper/figS2_soft_penalty.py

Last modified 2026-10-28
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.paper.common import (CM, OLD, OUT_DIR, SRC, Registry, md_find, md_row, panel_label, ringed_scatter, save_fig,
                                  set_style_old, vlabel)

NAME = "figS2_soft_penalty"
U = SRC["u1"]
LAMS = ["0", "0.01", "0.03"]
SETS = [("rich", "### Soft gate, enriched scalars, selection step", "### Soft gate, enriched scalars, confirmation step", OLD["navy"], OLD["navy_l"], "Enriched scalars"),
        ("thin", "### Soft gate, thin scalars, selection step", "### Soft gate, thin scalars, confirmation step", OLD["red"], OLD["red_l"], "Thin scalars")]


# the two mean labels of the enriched confirmation step in the left panel are moved off the dotted line (the values are
# unchanged, only the position moves, and a short line links each label back to its mean bar)
LABEL_Y = {("a", "rich", "dest_gate"): 0.70, ("a", "rich", "outcome_gate"): 0.028}


def num(cell):
    try:
        return float(cell)
    except ValueError:
        return None


def main():
    R = Registry(NAME)
    ex, cf = {}, {}
    rows = []
    for key, h_ex, h_cf, *_ in SETS:
        for lam in LAMS:
            ln = md_find(U, h_ex, 0, lam=lam)
            row = md_row(U, ln)
            for col, nm in [("outcome_gate_avg", "outcome_gate"), ("g_dest", "dest_gate"), ("drop_all_outcome_avg", "drop_all"),
                            ("gate_zero_outcome_avg", "gate_zero")]:
                v = num(row[col])
                if v is None or np.isnan(v):
                    continue
                R.md(f"{key}/explore/lam{lam}/{nm}", U, ln, col, fmt="f4")
                ex[(key, lam, nm)] = v
                rows.append(dict(setting=key, step="explore", seed=0, lam=lam, quantity=nm, value=v))
        for sd in [1, 2, 3]:
            ln = md_find(U, h_cf, 0, seed=sd)
            for col, nm in [("outcome_gate_avg", "outcome_gate"), ("dest_gate", "dest_gate"), ("drop_all_outcome_avg", "drop_all"),
                            ("gate_zero_outcome_avg", "gate_zero")]:
                v = R.md(f"{key}/confirm/s{sd}/{nm}", U, ln, col, fmt="f4")
                cf[(key, sd, nm)] = v
                rows.append(dict(setting=key, step="confirm", seed=sd, lam="0.01", quantity=nm, value=v))
        for nm in ["outcome_gate", "dest_gate", "drop_all", "gate_zero"]:
            R.derived(f"{key}/confirm/mean/{nm}", float(np.mean([cf[(key, sd, nm)] for sd in [1, 2, 3]])), "mean of these seeds",
                      [f"{key}/confirm/s{sd}/{nm}" for sd in [1, 2, 3]], fmt="f3")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{NAME}_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    set_style_old()
    fig, axes = plt.subplots(1, 2, figsize=(17.5 * CM, 7.0 * CM))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.88, wspace=0.22)
    xs = np.arange(len(LAMS))
    panels = [(axes[0], [("outcome_gate", "-", "o"), ("dest_gate", (0, (4, 2)), "s")], "Gate values", "Mean gate value", 0.10, "a"),
              (axes[1], [("drop_all", "-", "o"), ("gate_zero", (0, (4, 2)), "s")], "Loss without spatial input", "Mean AUC loss, outcome tasks", 0.01, "b")]
    for ax, quants, title, ylab, thr, let in panels:
        ax.grid(axis="y", color=OLD["grid"], lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(thr, color="#5F5F5F", lw=0.9, ls=(0, (1, 1.6)), zorder=1)
        for si, (key, _, _, col, lcol, lab) in enumerate(SETS):
            for qi, (q, ls, mk) in enumerate(quants):
                pts = [(i, ex[(key, lam, q)]) for i, lam in enumerate(LAMS) if (key, lam, q) in ex]
                if len(pts) > 1:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.4, ls=ls, zorder=2)
                for x, v in pts:
                    ringed_scatter(ax, [x], [v], col, s=26, marker=mk)
                c = np.array([cf[(key, sd, q)] for sd in [1, 2, 3]])
                xo = 1 + (0.17 if si == 0 else -0.17)
                ax.scatter(np.full(3, xo), c, s=12, color=lcol, marker=mk, edgecolor="white", linewidth=0.3, zorder=3)
                ax.plot([xo - 0.06, xo + 0.06], [c.mean()] * 2, color=col, lw=1.6, zorder=4)
                ty = LABEL_Y.get((let, key, q), c.mean())
                if ty != c.mean():
                    ax.plot([xo + 0.06, xo + 0.12], [c.mean(), ty], color=col, lw=0.6, zorder=4)
                vlabel(ax, xo + (0.13 if si == 0 else -0.09), ty, c.mean(), 3, ha="left" if si == 0 else "right", va="center",
                       fontsize=7.5, fontweight="bold", color=col, zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.9))
        ax.set_xticks(xs)
        ax.set_xticklabels(LAMS)
        ax.set_xlim(-0.35, 2.35)
        ax.set_xlabel("Opening penalty $\\lambda$")
        ax.set_ylabel(ylab)
        panel_label(ax, let, title)
    axes[0].text(2.33, 0.10, "0.10", ha="right", va="bottom", fontsize=7.5, color=OLD["gray"])
    axes[1].text(2.33, 0.01, "0.01", ha="right", va="bottom", fontsize=7.5, color=OLD["gray"])
    h0 = [Line2D([], [], color=OLD["navy"], lw=1.4, label="Enriched, seed 0"),
          Line2D([], [], color=OLD["red"], lw=1.4, label="Thin, seed 0"),
          Line2D([], [], color=OLD["ink"], lw=1.2, marker="o", ms=4, label="Outcome gate mean"),
          Line2D([], [], color=OLD["ink"], lw=1.2, ls=(0, (4, 2)), marker="s", ms=4, label="Destination gate"),
          Line2D([], [], marker="o", ls="", color=OLD["lgray"], ms=3.5, label="Seeds 1 to 3, bar = mean")]
    axes[0].legend(handles=h0, loc="lower right", bbox_to_anchor=(1.0, 0.17), borderaxespad=0.2, handlelength=1.8, labelspacing=0.28, handletextpad=0.4)
    h1 = [Line2D([], [], color=OLD["ink"], lw=1.2, marker="o", ms=4, label="Full tensor shuffled"),
          Line2D([], [], color=OLD["ink"], lw=1.2, ls=(0, (4, 2)), marker="s", ms=4, label="Gate forced closed")]
    axes[1].legend(handles=h1, loc="upper right", borderaxespad=0.2, handlelength=2.2, labelspacing=0.3)
    save_fig(fig, NAME)
    R.save(extra=dict(data_csv=str(OUT_DIR / f"{NAME}_data.csv")))
    print(f"[{NAME}] done, {len(rows)} numbers")


if __name__ == "__main__":
    main()
