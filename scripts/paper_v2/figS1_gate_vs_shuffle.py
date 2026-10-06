#!/usr/bin/env python3
"""
补充图 figS1_gate_vs_shuffle  任务级门控均值对“测试时打乱球员通道”的损失，基准统一模型、加厚统一模型、两通道变体三组
============================
横轴是各任务在 202 场测试事件上的门控均值（逐种子从逐事件文件重算，和 u1_clean_summary.md 各运行表的 gate_mean 核对），
纵轴是只打乱球员两个通道后的指标下降（AUC，落点是 skill），取各运行的 *_ablation.json 的 drop_players，和汇总表
“门控均值对球员通道消融下降量”一节的 drop_* 四位小数核对。每组的逐种子 Spearman 取同一节的 spearman_gate_vs_player_drop，
并用重算的门控和下降量再算一遍核对。基准统一模型只有种子 3 存了权重，所以只有一个种子。

输入  data/cache/u1_clean/{u1_s3,u1rich_s0..3,u1_2ch_s0..2}{.npz,_ablation.json}，audit/u1_clean_summary.md
输出  outputs/figures/v2/figS1_gate_vs_shuffle.{pdf,png}，scripts/paper_v2/out/figS1_gate_vs_shuffle_{sources.json,data.csv,labels.csv}

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/paper_v2/figS1_gate_vs_shuffle.py

Last modified 2026-10-04
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from scripts.paper_v2.common import (CM, DEV, OLD, OUT_DIR, SRC, TASK_NAME, TASKS, Registry, md_find, md_value, panel_label, rel,
                                     ringed_scatter, save_fig, set_style_old, unified_seed_values, vlabel)

NAME = "figS1_gate_vs_shuffle"
U = SRC["u1"]
GROUPS = [("u1", [3], "Unified model, seed 3"), ("u1rich", [0, 1, 2, 3], "Enriched unified, 4 seeds"),
          ("u1_2ch", [0, 1, 2], "Two-channel, 3 seeds")]
# 标签偏移（点），挤在一起的任务拉开并画引线
OFFS = {("u1", "dribble"): (-6, 4, "right"), ("u1", "tackle"): (6, 5, "left"), ("u1", "interception"): (6, -4, "left"),
        ("u1", "dest"): (0, 10, "center"), ("u1", "ball_recovery"): (0, 9, "center"),
        ("u1rich", "pass"): (6, -3, "left"), ("u1rich", "ball_recovery"): (6, -9, "left"), ("u1rich", "dribble"): (-6, 5, "right"),
        ("u1rich", "pressure"): (-6, 3, "right"), ("u1rich", "tackle"): (14, 14, "left"), ("u1rich", "interception"): (24, -1, "left"),
        ("u1rich", "shot"): (10, -8, "left"), ("u1rich", "dest"): (0, 10, "center"),
        ("u1_2ch", "pass"): (-8, 12, "right"), ("u1_2ch", "dribble"): (-9, 0, "right"), ("u1_2ch", "ball_recovery"): (8, -24, "left"), ("u1_2ch", "pressure"): (0, 10, "center"),
        ("u1_2ch", "interception"): (26, 4, "left"), ("u1_2ch", "tackle"): (26, -9, "left"), ("u1_2ch", "shot"): (20, 9, "left"),
        ("u1_2ch", "dest"): (7, 0, "left")}
SHORT = {"pass": "Pass", "dest": "Destination", "shot": "Shot", "interception": "Interception", "ball_recovery": "Ball recovery",
         "pressure": "Pressure", "dribble": "Dribble", "tackle": "Tackle"}


def main():
    R = Registry(NAME)
    data = {}
    rows = []
    for tag, seeds, _ in GROUPS:
        for sd in seeds:
            abl_f = DEV / "data" / "cache" / "u1_clean" / f"{tag}_s{sd}_ablation.json"
            abl = json.load(open(abl_f))
            ln = md_find(U, "## 门控均值对球员通道消融下降量", 0, run=tag, seed=sd)
            gates, drops = [], []
            for t in TASKS:
                gv = unified_seed_values(tag, t, "gate")[sd]
                dv = abl[t]["drop_players"]
                assert round(dv, 4) == round(md_value(U, ln, f"drop_{t}"), 4) or abs(round(dv, 4)) == abs(md_value(U, ln, f"drop_{t}")), f"{tag} s{sd} {t}"
                R.perevent(f"{tag}/s{sd}/{t}/gate", gv, f"{tag}_s{sd}.npz 的 gate_mean 在 {t} 的 202 场测试事件上取平均", fmt="f4")
                R.add(f"{tag}/s{sd}/{t}/drop", dv, dict(kind="json", file=rel(abl_f), path=[t, "drop_players"]), fmt="f4")
                gates.append(gv)
                drops.append(dv)
                rows.append(dict(run=tag, seed=sd, task=t, gate=round(gv, 6), drop_players=round(dv, 6)))
            rho = spearmanr(gates, drops).statistic
            rmd = R.md(f"{tag}/s{sd}/spearman", U, ln, "spearman_gate_vs_player_drop", fmt="f3")
            assert round(rho, 3) == round(rmd, 3), f"{tag} s{sd} Spearman 重算 {rho:.3f} 汇总表 {rmd}"
            data[(tag, sd)] = (np.array(gates), np.array(drops), rmd)
        # 种子平均的门控和运行表核对
        for t in TASKS:
            lt = md_find(U, f"## 运行 {tag}（", 0, task=t)
            gm = float(np.mean([data[(tag, s)][0][TASKS.index(t)] for s in seeds]))
            if len(seeds) == len(unified_seed_values(tag, t, "gate")):
                assert round(gm, 4) == round(md_value(U, lt, "gate_mean"), 4), f"{tag} {t} 门控均值"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{NAME}_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    set_style_old()
    fig, axes = plt.subplots(1, 3, figsize=(17.5 * CM, 6.6 * CM), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.17, top=0.86, wspace=0.07)
    rng = np.random.RandomState(3)
    for ax, (tag, seeds, title), let in zip(axes, GROUPS, "abc"):
        ax.grid(color=OLD["grid"], lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        G = np.array([data[(tag, s)][0] for s in seeds])
        D = np.array([data[(tag, s)][1] for s in seeds])
        for i, t in enumerate(TASKS):
            col, lcol, mk = (OLD["red"], OLD["red_l"], "s") if t == "dest" else (OLD["navy"], OLD["navy_l"], "o")
            if len(seeds) > 1:
                ax.scatter(G[:, i], D[:, i], s=10, color=lcol, marker=mk, edgecolor="white", linewidth=0.3, zorder=2)
            gm, dm = G[:, i].mean(), D[:, i].mean()
            ringed_scatter(ax, [gm], [dm], col, s=30, marker=mk)
            dx, dy, ha = OFFS.get((tag, t), (5, 2, "left"))
            ap = dict(arrowstyle="-", lw=0.45, color=OLD["gray"], shrinkA=0, shrinkB=3) if abs(dx) + abs(dy) > 14 else None
            ax.annotate(SHORT[t], (gm, dm), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=OLD["ink"], zorder=6,
                        ha=ha, va="center", arrowprops=ap)
        rho = [data[(tag, s)][2] for s in seeds]
        ax.text(0.03, 0.97, "Spearman per seed\n" + ", ".join(f"{r:.3f}".replace("-", "−") for r in rho), transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5, color=OLD["ink"], linespacing=1.25)
        for r in rho:
            from scripts.paper_v2.common import VLABELS
            VLABELS.append((f"{r:.3f}", float(r)))
        ax.set_yscale("log")
        ax.set_xlim(0.05, 0.92)
        ax.set_xlabel("Task gate mean")
        panel_label(ax, let, title)
    axes[0].set_ylabel("Loss when player channels are shuffled")
    h = [Line2D([], [], marker="o", ls="", color=OLD["navy"], markeredgecolor="white", ms=5.5, label="Outcome task, AUC loss"),
         Line2D([], [], marker="s", ls="", color=OLD["red"], markeredgecolor="white", ms=5.5, label="Destination, skill loss"),
         Line2D([], [], marker="o", ls="", color=OLD["navy_l"], markeredgecolor="white", ms=3.5, label="One seed (b, c)")]
    axes[2].legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 0.0), borderaxespad=0.2, handletextpad=0.3, labelspacing=0.3)
    save_fig(fig, NAME)
    R.save(extra=dict(data_csv=str(OUT_DIR / f"{NAME}_data.csv")))
    print(f"[{NAME}] 画好，{len(rows)} 个点")


if __name__ == "__main__":
    main()
