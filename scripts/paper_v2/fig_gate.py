#!/usr/bin/env python3
"""
图 fig_gate  统一模型八个任务的门控均值（四个种子），叠上落点干预和 Tackle 干预重训后的门控（各三个种子）
============================
左图是落点干预，右图是 Tackle 干预，灰色是不干预的统一模型 u1（种子 0 到 3），彩色是干预运行（种子 0 到 2），短横线是种子均值。
逐种子门控从逐事件文件重算（summarize_u1_clean.load_runs，202 场），均值和 u1_clean_summary.md 各运行表的 gate_mean 核对，
种子 0 到 2 的逐种子值另和“干预”一节的 gate_base、gate_shuffled_run 核对（四舍五入到四位）。

输入  data/cache/u1_clean/{u1,shuf_dest,shuf_tackle}_s*.npz，audit/u1_clean_summary.md（核对）
输出  outputs/figures/v2/fig_gate.{pdf,png}，scripts/paper_v2/out/fig_gate_data.csv 和 _sources.json

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/paper_v2/fig_gate.py

Last modified 2026-10-04（16:30 起版式按 5 月旧图重做，数值未变）
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.paper_v2.common import (C, CM, OLD, OUT_DIR, SRC, TASK_NAME, TASKS, Registry, md_find, md_value, panel_label,
                                     ringed_scatter, save_fig, set_style_old, unified_seed_values, vlabel)

NAME = "fig_gate"
U = SRC["u1"]
PANELS = [("shuf_dest", "dest", "Destination tensors shuffled"), ("shuf_tackle", "tackle", "Tackle tensors shuffled")]


def seed_gates(R, tag):
    out = {}
    for t in TASKS:
        per = unified_seed_values(tag, t, "gate")
        ln = md_find(U, f"## 运行 {tag}（", 0, task=t)
        ref = md_value(U, ln, "gate_mean")
        assert round(float(np.mean(list(per.values()))), 4) == round(ref, 4), f"{tag} {t} 门控均值重算和汇总表不一致"
        R.md(f"{tag}/{t}/gate_mean", U, ln, "gate_mean", fmt="f4")
        for s, v in per.items():
            R.perevent(f"{tag}/{t}/s{s}", v, f"{tag}_s{s}.npz 的 gate_mean 在 {t} 的 202 场测试事件上取平均", fmt="f4")
        out[t] = per
    return out


def main():
    R = Registry(NAME)
    base = seed_gates(R, "u1")
    shuf = {tag: seed_gates(R, tag) for tag, _, _ in PANELS}
    # 和干预表逐种子核对
    for tag, task, _ in PANELS:
        for s in [0, 1, 2]:
            for t in TASKS:
                ln = md_find(U, "## 干预", 0, run=tag, seed=s, task=t)
                assert round(base[t][s], 4) == round(md_value(U, ln, "gate_base"), 4)
                assert round(shuf[tag][t][s], 4) == round(md_value(U, ln, "gate_shuffled_run"), 4)
        R.notes.append(f"{tag} 种子 0 到 2 的八个任务门控和干预表的 gate_base、gate_shuffled_run 四位一致")
    rows = [dict(run="u1", task=t, seed=s, gate=round(v, 6)) for t in TASKS for s, v in base[t].items()]
    rows += [dict(run=tag, task=t, seed=s, gate=round(v, 6)) for tag in shuf for t in TASKS for s, v in shuf[tag][t].items()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{NAME}_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    set_style_old()
    order = sorted(TASKS, key=lambda t: -np.mean(list(base[t].values())))
    fig, axes = plt.subplots(1, 2, figsize=(17.5 * CM, 7.6 * CM), sharey=True)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.13, top=0.9, wspace=0.06)
    y = np.arange(len(order))[::-1].astype(float)
    rng = np.random.RandomState(1)
    lab = lambda v: f"{v:.3f}"
    for ax, (tag, task, title), let in zip(axes, PANELS, "ab"):
        for yy in y:
            ax.axhline(yy, color=OLD["grid"], lw=0.6, zorder=0)
        i0 = order.index(task)
        ax.axhspan(y[i0] - 0.45, y[i0] + 0.45, color=OLD["band"], zorder=0, lw=0)
        for i, t in enumerate(order):
            b = np.array(list(base[t].values()))
            a = np.array(list(shuf[tag][t].values()))
            bm, am = b.mean(), a.mean()
            ax.plot([bm, am], [y[i], y[i]], color=OLD["stem"], lw=2.6, solid_capstyle="round", zorder=1)
            ax.scatter(b, np.full(len(b), y[i] + 0.2) + (rng.rand(len(b)) - 0.5) * 0.06, s=9, color=OLD["navy_l"],
                       edgecolor="white", linewidth=0.3, zorder=2)
            ax.scatter(a, np.full(len(a), y[i] - 0.2) + (rng.rand(len(a)) - 0.5) * 0.06, s=9, color=OLD["red_l"],
                       edgecolor="white", linewidth=0.3, zorder=2)
            ringed_scatter(ax, [bm], [y[i]], OLD["navy"], s=44)
            ringed_scatter(ax, [am], [y[i]], OLD["red"], s=44)
            left, right = (bm, am) if bm <= am else (am, bm)
            lcol = OLD["navy"] if bm <= am else OLD["red"]
            rcol = OLD["red"] if bm <= am else OLD["navy"]
            vlabel(ax, left - 0.022, y[i], left, 3, ha="right", va="center", fontsize=7.3, fontweight="bold", color=lcol, zorder=5)
            vlabel(ax, right + 0.022, y[i], right, 3, ha="left", va="center", fontsize=7.3, fontweight="bold", color=rcol, zorder=5)
        rk = []
        for sd in [0, 1, 2]:
            ln = md_find(U, "## 干预", 0, run=tag, seed=sd, task=task)
            rk.append((int(md_value(U, ln, "rank_base")), int(md_value(U, ln, "rank_shuffled_run"))))
            R.md(f"{tag}/rank_base/s{sd}", U, ln, "rank_base", fmt="int")
            R.md(f"{tag}/rank_after/s{sd}", U, ln, "rank_shuffled_run", fmt="int")
        rtxt = ("1 to 8 in all three seeds" if all(r == (1, 8) for r in rk) else ", ".join(f"{p} to {q}" for p, q in rk) + "\nin seeds 0, 1, 2")
        ax.text(0.985, 0.2 if task == "dest" else 0.56, f"{TASK_NAME[task]} gate rank among 8\n{rtxt}",
                transform=ax.transAxes, ha="right", va="center", fontsize=7.5, color=OLD["ink"], linespacing=1.3)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Mean gate value (weight on the spatial encoding)")
        panel_label(ax, let, title)
        R.notes.append(f"{tag} 名次（基准到干预）{rk}")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([TASK_NAME[t] for t in order])
    h = [Line2D([], [], marker="o", ls="", color=OLD["navy"], markeredgecolor="white", ms=6, label="Unified model, 4-seed mean"),
         Line2D([], [], marker="o", ls="", color=OLD["red"], markeredgecolor="white", ms=6, label="After shuffling, 3-seed mean"),
         Line2D([], [], marker="o", ls="", color=OLD["lgray"], markeredgecolor="white", ms=3.6, label="One seed")]
    axes[1].legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.0), borderaxespad=0.2, handletextpad=0.4, labelspacing=0.3)
    save_fig(fig, NAME)
    R.save(extra=dict(data_csv=str(OUT_DIR / f"{NAME}_data.csv")))
    print(f"[{NAME}] 画好，{len(rows)} 个点")


if __name__ == "__main__":
    main()
