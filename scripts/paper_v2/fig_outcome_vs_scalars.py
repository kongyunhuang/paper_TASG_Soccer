#!/usr/bin/env python3
"""
图 fig_outcome_vs_scalars  七个结果类任务上 CNN 相对同一标量集 XGBoost 的 AUC 差，薄标量和加厚标量两组并排
============================
每个点是一个种子的配对差（同种子的 M4_CNN_Full 减 B2_XGB_360，202 场），薄标量五个种子，加厚标量三个种子（M4 只有三个）。
逐种子值从同数据单任务的逐事件预测重算（口径照搬 summarize_u1_clean.single_task_table），再和 u1_clean_summary.md
两张“同种子配对差”表的 mean、sd、min、max 逐项核对，一致才画。0 线是同一标量集的 XGBoost，虚线是 +0.01。两组同一个 y 轴范围。

输入  data/cache/single_clean{,_rich}/preds/*.npz，audit/u1_clean_summary.md（核对）
输出  outputs/figures/v2/fig_outcome_vs_scalars.{pdf,png}，scripts/paper_v2/out/fig_outcome_vs_scalars_data.csv 和 _sources.json

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/paper_v2/fig_outcome_vs_scalars.py

Last modified 2026-10-04（16:30 起版式按 5 月旧图重做，数值未变）
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.paper_v2.common import (C, CM, OLD, OUT_DIR, OUTCOME, SRC, TASK_NAME, Registry, md_find, md_value, panel_label,
                                     ringed_scatter, save_fig, set_style_old, single_seed_values, vlabel)

NAME = "fig_outcome_vs_scalars"
GROUPS = [("single_clean", "Thin scalars", C["blue"], "## 同数据单任务模型（202"),
          ("single_clean_rich", "Enriched scalars", C["orange"], "## 同数据单任务模型，标量加厚")]


def main():
    R = Registry(NAME)
    data = {}
    rows = []
    for d, label, col, head in GROUPS:
        for t in OUTCOME:
            a = single_seed_values(d, t, "M4_CNN_Full")
            b = single_seed_values(d, t, "B2_XGB_360")
            seeds = sorted(set(a) & set(b))
            diff = np.array([a[s] - b[s] for s in seeds])
            # 和汇总表配对差一行逐项核对
            ln = md_find(SRC["u1"], head, 1, task=t, pair="M4_CNN_Full 减 B2_XGB_360")
            for k, v in [("mean", diff.mean()), ("sd", diff.std(ddof=1)), ("min", diff.min()), ("max", diff.max())]:
                mv = md_value(SRC["u1"], ln, k)
                assert round(v, 4) == round(mv, 4) or abs(round(v, 4) - mv) < 1e-9, f"{d} {t} {k} 重算 {v:.4f} 汇总表 {mv}"
                R.md(f"{d}/{t}/{k}", SRC["u1"], ln, k, fmt="f4")
            npos = int((diff > 0).sum())
            assert f"{npos}/{len(diff)}" == md_value(SRC["u1"], ln, "seeds_positive", "text"), f"{d} {t} 正的种子数不一致"
            for s, v in zip(seeds, diff):
                R.perevent(f"{d}/{t}/s{s}", v, f"{d}/preds/{t}_M4_CNN_Full_s{s}.npz 减 {t}_B2_XGB_360_s{s}.npz 的 AUC，202 场", fmt="f4")
                rows.append(dict(group=label, task=t, seed=s, cnn_minus_xgb=round(float(v), 6)))
            data[(d, t)] = (seeds, diff)
            R.notes.append(f"{d} {t} 逐种子差 {np.round(diff, 4).tolist()}，均值 {diff.mean():.4f}，和汇总表 L{ln} 一致")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{NAME}_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    set_style_old()
    fig, axes = plt.subplots(1, 2, figsize=(17.5 * CM, 7.4 * CM), sharey=True)
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.2, top=0.9, wspace=0.05)
    allv = np.concatenate([v for _, v in data.values()])
    lo, hi = allv.min() - 0.0035, allv.max() + 0.0035
    x = np.arange(len(OUTCOME))
    rng = np.random.RandomState(0)
    for ax, (d, label, _, _), let in zip(axes, GROUPS, "ab"):
        ax.grid(axis="y", color=OLD["grid"], lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(0, color="#5F5F5F", lw=0.9, ls=(0, (4, 2.5)), zorder=1)
        ax.axhline(0.01, color=OLD["lgray"], lw=0.9, ls=(0, (1, 1.6)), zorder=1)
        ax.text(len(OUTCOME) - 0.42, 0.0102, "+0.01", ha="right", va="bottom", fontsize=7.5, color=OLD["gray"])
        for i, t in enumerate(OUTCOME):
            seeds, diff = data[(d, t)]
            m = diff.mean()
            col, lcol = (OLD["navy"], OLD["navy_l"]) if m >= 0 else (OLD["red"], OLD["red_l"])
            ax.plot([x[i], x[i]], [0, m], color=col, lw=1.8, alpha=0.85, solid_capstyle="round", zorder=2)
            jit = (rng.rand(len(diff)) - 0.5) * 0.30
            ax.scatter(x[i] + jit, diff, s=11, color=lcol, edgecolor="white", linewidth=0.3, zorder=3)
            ringed_scatter(ax, [x[i]], [m], col, s=46)
            yl = (max(diff.max(), m) + 0.0012) if m >= 0 else (min(diff.min(), m) - 0.0012)
            vlabel(ax, x[i], yl, m, 4, ha="center", va="bottom" if m >= 0 else "top", fontsize=7.5, fontweight="bold", color=col, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_NAME[t] for t in OUTCOME], rotation=30, ha="right")
        ax.set_xlim(-0.6, len(OUTCOME) - 0.4)
        ax.set_ylim(lo, hi)
        ax.tick_params(axis="x", length=0)
        panel_label(ax, let, label)
    axes[0].set_ylabel("CNN minus XGBoost (same scalars), AUC")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=OLD["navy"], markeredgecolor="white", ms=6, label="Seed mean, CNN ahead"),
         Line2D([], [], marker="o", ls="", color=OLD["red"], markeredgecolor="white", ms=6, label="Seed mean, XGBoost ahead"),
         Line2D([], [], marker="o", ls="", color=OLD["navy_l"], markeredgecolor="white", ms=3.8, label="One seed, paired")]
    axes[1].legend(handles=h, loc="upper right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2, handletextpad=0.4, labelspacing=0.35)
    save_fig(fig, NAME)
    R.notes.append(f"y 轴范围 {lo:.4f} 到 {hi:.4f}，两图共用")
    R.save(extra=dict(data_csv=str(OUT_DIR / f"{NAME}_data.csv")))
    print(f"[{NAME}] 画好，{len(rows)} 个种子点")


if __name__ == "__main__":
    main()
