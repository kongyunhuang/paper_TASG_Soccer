#!/usr/bin/env python3
"""
图 fig_holdout  四条留出确认量的 24/25 对 25/26，25/26 带按比赛自助法 95% 区间，每行标出确认阈值
============================
C1、C2、C3 的量、区间和 24/25 参照取 audit/holdout_2526_summary.md（和 tab_holdout 同一批行），阈值取同一文件开头的阈值字典，
判定词取冻结方案文末判定表。C4 的 25/26 逐种子门控取 summary.json 的 derived.C4，24/25 逐种子门控从 u1_clean 逐事件文件重算
（均值和 u1_clean_summary.md 核对）。门控读数没有自助法区间。

输入  audit/holdout_2526_summary.md，data/cache/holdout_2526/summary.json，data/cache/u1_clean/{u1,shuf_dest}_s*.npz，audit/u1_clean_summary.md
输出  outputs/figures/v2/fig_holdout.{pdf,png}，scripts/paper_v2/out/fig_holdout_data.csv 和 _sources.json

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/paper_v2/fig_holdout.py

Last modified 2026-10-04（16:30 起版式按 5 月旧图重做，数值未变）
"""
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.paper_v2.common import (C, CM, OLD, OUT_DIR, SRC, Registry, json_value, md_find, md_row, md_value, rel, ringed_scatter,
                                     save_fig, set_style_old, unified_seed_values, vlabel)

NAME = "fig_holdout"
H, J, U, F = SRC["hold_md"], SRC["hold_json"], SRC["u1"], SRC["frozen"]


def num(v, nd):
    """标签里的数，负号用数学减号"""
    return f"{v:.{nd}f}".replace("-", "\u2212")


def thr_value(R, key, iid):
    L = H.read_text(encoding="utf8").split("\n")
    ln = next(i for i, x in enumerate(L) if f'"{key}"' in x) + 1
    v = float(re.search(r":\s*([-0-9.]+)", L[ln - 1]).group(1))
    R.add(iid, v, dict(kind="mdline", file=rel(H), line=ln, regex=r":\s*([-0-9.]+)"), fmt="f3")
    return v


def main():
    R = Registry(NAME)
    panels = []   # (标题, 行列表, x 轴标签)；行 = dict(label, v25, v26, lo, hi, thr, kind)

    # C1
    rows = []
    for label, iid, key, txt in [("u1fcn 减最强规则", "C1/fcn_minus_rule", "fcn_minus_rule_confirm", "Variant minus rule\n"),
                                 ("u1fcn 减 M2 加厚", "C1/fcn_minus_m2rich", "fcn_minus_m2rich_confirm", "Variant minus\nenriched MLP ")]:
        ln = md_find(H, "## C1", 0, **{"量": label})
        t = thr_value(R, key, iid + "/thr")
        rows.append(dict(label=f"{txt}(≥ {num(t, 3)})", v25=R.md(iid + "/2425", H, ln, "24/25"), v26=R.md(iid + "/2526", H, ln, "留出集"),
                         lo=R.md(iid + "/lo", H, ln, "95% 区间", "lo"), hi=R.md(iid + "/hi", H, ln, "95% 区间", "hi"), thr=[t]))
    panels.append(("C1", "Pass destination, spatial structure cannot be replaced", rows, "Difference in top-1 accuracy"))

    # C2
    rows = []
    for md_label, col, iid, key, txt, op in [
            ("XGB 加厚减 M4 薄 留出集", "三任务平均", "C2/a", "xgbrich_minus_m4thin_confirm", "Enriched XGBoost\nminus thin CNN", "≥"),
            ("厚增益 M4 减 M2 留出集", "三任务平均", "C2/b", "rich_gain_abs_confirm", "Enriched gain\n", "≤"),
            ("薄增益 M4 减 M2 留出集", "三任务平均", "C2/thin", None, "Gain, thin scalars\n(reference for enriched gain)", None),
            ("薄增益 M4 减 M2 留出集", "pressure", "C2/c_thin", "pressure_thin_gain_min", "Pressure gain,\nthin", "≥"),
            ("厚增益 M4 减 M2 留出集", "pressure", "C2/c_rich", "pressure_rich_gain_max", "Pressure enriched\ngain", "≤")]:
        ln = md_find(H, "## C2", 0, **{"量": md_label})
        assert md_row(H, ln + 1)["量"] == "同上 95% 区间" and md_row(H, ln + 2)["量"] == "同上 24/25"
        t = thr_value(R, key, iid + "/thr") if key else None
        lab = f"{txt} ({op} {num(t, 3)})" if key else txt
        rows.append(dict(label=lab, v25=R.md(iid + "/2425", H, ln + 2, col), v26=R.md(iid + "/2526", H, ln, col),
                         lo=R.md(iid + "/lo", H, ln + 1, col, "lo"), hi=R.md(iid + "/hi", H, ln + 1, col, "hi"), thr=[t] if key else []))
    panels.append(("C2", "Outcome tasks, enriched scalars replace most of the spatial gain", rows, "Difference in AUC, mean over Pass, Ball recovery and Pressure unless Pressure is named"))

    # C3
    rows = []
    ln = md_find(H, "## C3", 0, **{"量": "u1 减 XGB 加厚 留出集"})
    for col, iid, key, txt in [("pass", "C3/pass", "pass_abs_confirm", "Pass"), ("pressure", "C3/pressure", "pressure_abs_confirm", "Pressure"),
                               ("ball_recovery", "C3/br", "br_abs_confirm", "Ball recovery\n"),
                               ("三任务平均（Pass BR Pressure）", "C3/mean3", "mean3_abs_confirm", "Mean of three\n")]:
        t = thr_value(R, key, iid + "/thr")
        rows.append(dict(label=f"{txt}{'' if txt.endswith(chr(10)) else ' '}(|d| ≤ {num(t, 3)})", v25=R.md(iid + "/2425", H, ln + 2, col), v26=R.md(iid + "/2526", H, ln, col),
                         lo=R.md(iid + "/lo", H, ln + 1, col, "lo"), hi=R.md(iid + "/hi", H, ln + 1, col, "hi"), thr=[-t, t], band=True))
    panels.append(("C3", "Unified model stays close to enriched XGBoost", rows, "Unified model minus enriched XGBoost, AUC"))

    # C4
    rows = []
    for tag, key, txt, op in [("u1", "u1_dest_gate_min", "Unified model\n", "≥"),
                              ("shuf_dest", "shuf_dest_gate_max", "Destination shuffled\n", "≤")]:
        per25 = unified_seed_values(tag, "dest", "gate")
        lu = md_find(U, f"## 运行 {tag}（", 0, task="dest")
        assert round(float(np.mean(list(per25.values()))), 4) == round(md_value(U, lu, "gate_mean"), 4)
        R.md(f"C4/{tag}/2425_mean", U, lu, "gate_mean", fmt="f4")
        for s, v in per25.items():
            R.perevent(f"C4/{tag}/2425/s{s}", v, f"{tag}_s{s}.npz 落点事件的 gate_mean 平均，202 场", fmt="f4")
        seeds = sorted(json_value(J, ["derived", "C4", tag]).keys())
        per26 = {int(s): R.js(f"C4/{tag}/2526/s{s}", J, ["derived", "C4", tag, s, "dest_gate"], fmt="f4") for s in seeds}
        t = thr_value(R, key, f"C4/{tag}/thr")
        m25 = next(it["value"] for it in R.items if it["id"] == f"C4/{tag}/2425_mean")
        m26 = R.derived(f"C4/{tag}/2526_mean", float(np.mean(list(per26.values()))), "这几个种子的均值", [f"C4/{tag}/2526/s{s}" for s in seeds], fmt="f3")
        rows.append(dict(label=f"{txt}({op} {num(t, 2)})", s25=list(per25.values()), s26=list(per26.values()), thr=[t], m25=m25, m26=m26))
    panels.append(("C4", "Destination gate pattern on new matches, gate reading only", rows, "Mean gate value of the destination task, one point per seed"))

    # 判定词
    FL = F.read_text(encoding="utf8").split("\n")
    v0 = next(i for i, x in enumerate(FL) if x.startswith("### 判定（按第二节的阈值）"))
    words = {"C1": ("confirmed", "**确认**"), "C2": ("confirmed, Pressure near the bound", "**确认**"), "C3": ("inconclusive", "**不确定**"), "C4": ("holds", "**保持**")}
    verdict = {}
    for c, (w, must) in words.items():
        ln = next(i for i, x in enumerate(FL) if i > v0 and x.startswith(f"| {c} ")) + 1
        verdict[c] = R.text(f"{c}/verdict", w, F, ln, must)

    # 表格视图
    out = []
    for c, title, rows, _ in panels:
        for r in rows:
            if "s25" in r:
                out += [dict(claim=c, quantity=r["label"], season="2425", seed=k, value=v) for k, v in enumerate(r["s25"])]
                out += [dict(claim=c, quantity=r["label"], season="2526", seed=k, value=v) for k, v in enumerate(r["s26"])]
            else:
                out += [dict(claim=c, quantity=r["label"], season="2425", seed="", value=r["v25"]),
                        dict(claim=c, quantity=r["label"], season="2526", seed="", value=r["v26"], lo=r["lo"], hi=r["hi"])]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{NAME}_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["claim", "quantity", "season", "seed", "value", "lo", "hi"])
        w.writeheader()
        w.writerows(out)

    set_style_old()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(17.5 * CM, 10.6 * CM))
    nrow = {c: len(rows) for c, _, rows, _ in panels}
    # 左列 C1、C3，右列 C2、C4，两列各自按行数分高度
    gl = GridSpec(2, 1, figure=fig, left=0.19, right=0.43, top=0.92, bottom=0.1, hspace=0.62,
                  height_ratios=[nrow["C1"] + 0.9, nrow["C3"] + 0.9])
    gr = GridSpec(2, 1, figure=fig, left=0.705, right=0.935, top=0.92, bottom=0.1, hspace=0.62,
                  height_ratios=[nrow["C2"] + 0.9, nrow["C4"] + 0.9])
    axd = {"C1": fig.add_subplot(gl[0]), "C3": fig.add_subplot(gl[1]), "C2": fig.add_subplot(gr[0]), "C4": fig.add_subplot(gr[1])}
    short = {"C1": "C1, pass destination", "C2": "C2, outcome tasks", "C3": "C3, unified model", "C4": "C4, destination gate"}
    xlabs = {"C1": "Difference in top-1 accuracy", "C2": "AUC difference, mean of Pass,\nBall recovery, Pressure unless named",
             "C3": "Unified minus enriched XGBoost, AUC", "C4": "Destination gate value, points are seeds"}
    fmt = lambda v, nd: f"{v:.{nd}f}".replace("-", "\u2212")
    for (c, title, rows, xlab), let in zip(panels, "abcd"):
        ax = axd[c]
        n = len(rows)
        y = np.arange(n)[::-1].astype(float)
        for r, yy in zip(rows, y):
            ax.axhline(yy, color=OLD["grid"], lw=0.6, zorder=0)
            if r.get("band"):
                ax.fill_betweenx([yy - 0.36, yy + 0.36], r["thr"][0], r["thr"][1], color=OLD["band"], lw=0, zorder=0)
            else:
                for t in r["thr"]:
                    ax.plot([t, t], [yy - 0.36, yy + 0.36], color="#5F5F5F", lw=1.0, ls=(0, (2.5, 1.5)), zorder=2)
            if "s25" in r:
                ringed_scatter(ax, r["s25"], np.full(len(r["s25"]), yy + 0.15), OLD["navy"], s=24)
                ringed_scatter(ax, r["s26"], np.full(len(r["s26"]), yy - 0.15), OLD["red"], s=24, marker="s")
                a, b, nd = r["m25"], r["m26"], 3
            else:
                ax.plot([r["lo"], r["hi"]], [yy - 0.15] * 2, color=OLD["red"], lw=1.6, solid_capstyle="round", zorder=3)
                ringed_scatter(ax, [r["v25"]], [yy + 0.15], OLD["navy"], s=34)
                ringed_scatter(ax, [r["v26"]], [yy - 0.15], OLD["red"], s=34, marker="s")
                a, b, nd = r["v25"], r["v26"], 4
            vlabel(ax, 1.03, yy + 0.15, a, nd, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7.5,
                   fontweight="bold", color=OLD["navy"])
            vlabel(ax, 1.03, yy - 0.15, b, nd, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7.5,
                   fontweight="bold", color=OLD["red"])
        if c in ("C2", "C3"):
            ax.axvline(0, color="#5F5F5F", lw=0.8, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([r["label"] for r in rows], fontsize=7.5, linespacing=1.05)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.55, n - 0.45)
        ax.set_xlabel(xlabs[c], fontsize=7.8, labelpad=2)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.text(-0.78, 1.06, f"({let}) {short[c]}", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold")
        ax.text(1.27, 1.06, verdict[c][0].upper() + verdict[c][1:], transform=ax.transAxes, ha="right", va="bottom", fontsize=7.8,
                color=OLD["ink"], style="italic")
    axd["C4"].set_xlim(0.15, 0.9)
    axd["C1"].set_xlim(0.025, 0.122)
    h = [Line2D([], [], marker="o", ls="", color=OLD["navy"], ms=5, markeredgecolor="white", label="2024/25"),
         Line2D([], [], marker="s", ls="-", color=OLD["red"], ms=5, markeredgecolor="white", lw=1.6, label="2025/26, 95% CI"),
         Line2D([], [], color="#5F5F5F", lw=1.0, ls=(0, (2.5, 1.5)), label="Threshold")]
    axd["C2"].legend(handles=h, loc="upper right", bbox_to_anchor=(1.0, 1.03), borderaxespad=0.1, handlelength=1.5, labelspacing=0.3,
                     fontsize=7.3, handletextpad=0.4)
    save_fig(fig, NAME)
    R.save(extra=dict(data_csv=str(OUT_DIR / f"{NAME}_data.csv")))
    print(f"[{NAME}] 画好，{len(out)} 个点")


if __name__ == "__main__":
    main()
