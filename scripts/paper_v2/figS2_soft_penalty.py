#!/usr/bin/env python3
"""
补充图 figS2_soft_penalty  软门控加开启代价的探索步（种子 0 沿 λ 的路径）和确认步（λ = 0.01，种子 1、2、3）
============================
左图  结果类任务门控平均和落点门控，标量加厚和标量薄两套，各一条种子 0 的线，确认步三个种子画成点并标均值。
右图  整张量打乱后结果类任务的平均损失和测试时强制关门的平均损失；标量薄的探索步只有 λ = 0.01 做过这两种消融。
数全部取 audit/u1_clean_summary.md 的“门控代价实验”一节（软门控加厚探索步、确认步，软门控薄探索步、确认步四张表），按行号登记。
虚线是预先写下的标准里甲的两条阈值（结果类门控平均不超过 0.10，整张量打乱损失不超过 0.01）。

输入  audit/u1_clean_summary.md
输出  outputs/figures/v2/figS2_soft_penalty.{pdf,png}，scripts/paper_v2/out/figS2_soft_penalty_{sources.json,data.csv,labels.csv}

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/paper_v2/figS2_soft_penalty.py

Last modified 2026-10-04
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.paper_v2.common import (CM, OLD, OUT_DIR, SRC, Registry, md_find, md_row, panel_label, ringed_scatter, save_fig,
                                     set_style_old, vlabel)

NAME = "figS2_soft_penalty"
U = SRC["u1"]
LAMS = ["0", "0.01", "0.03"]
SETS = [("rich", "### 软门控，标量加厚，探索步", "### 软门控，标量加厚，确认步", OLD["navy"], OLD["navy_l"], "Enriched scalars"),
        ("thin", "### 软门控，标量薄，探索步", "### 软门控，标量薄，确认步", OLD["red"], OLD["red_l"], "Thin scalars")]


# 左图加厚确认步的两个均值标签挪离虚线（数值不变，只挪位置，用短线连回均值横线）
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
            R.derived(f"{key}/confirm/mean/{nm}", float(np.mean([cf[(key, sd, nm)] for sd in [1, 2, 3]])), "这几个种子的均值",
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
    print(f"[{NAME}] 画好，{len(rows)} 个数")


if __name__ == "__main__":
    main()
