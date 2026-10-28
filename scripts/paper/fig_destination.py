#!/usr/bin/env python3
"""
fig_destination.py, ladder of methods on the pass destination task
==================================================================
Strongest rule, MLP with thin and with enriched scalars, unified model with the pooled encoder and resolution-preserving
variant, with 2024/25 and 2025/26 side by side. Left panel top-1, right panel skill (1 minus log loss over ln96). Bars are
seed means with the seed SD; the rule has no randomness and has a single value.
The per-seed values of 2024/25 are recomputed from the per-event files (single-task models as in
summarize_u1_clean.single_task_table, unified models with load_runs) and their means checked against
results/u1_clean_summary.md. The per-seed values of 2025/26 are metrics.*.by_seed of results/holdout_2526_summary.json,
with their mean checked against its mean. The rule comes from results/dest_rule_baselines.md and
results/holdout_2526_dest_rules.json.

Inputs   data/cache/u1_clean/*.npz, data/cache/single_clean{,_rich}/preds/dest_*.npz, results/holdout_2526_summary.json,
         results/holdout_2526_dest_rules.json, results/dest_rule_baselines.md, results/u1_clean_summary.md
Outputs  outputs/figures/fig_destination.{pdf,png}, results/figure_data/fig_destination_{data,labels}.csv,
         outputs/provenance/fig_destination_sources.json

Usage
  PYTHONPATH=. python -u scripts/paper/fig_destination.py

Last modified 2026-10-28
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.paper.common import (C, CM, OLD, OUT_DIR, SRC, Registry, json_value, md_find, md_value, panel_label, save_fig,
                                  set_style_old, single_seed_values, unified_seed_values, vlabel)

NAME = "fig_destination"
RULE = "Table times teammates over opponents (a=2, b=0.5)"
METHODS = [("rule", None, None, "Strongest rule"),
           ("single", "single_clean", "M2_MLP_360", "MLP, thin scalars"),
           ("single", "single_clean_rich", "M2_MLP_360", "MLP, enriched scalars"),
           ("unified", "u1", None, "Unified model, pooled encoder"),
           ("unified", "u1fcn", None, "Resolution-preserving variant")]
HEAD = {"single_clean": "## Single-task models on the same data (202", "single_clean_rich": "## Single-task models on the same data, enriched scalars"}
HKEY = {"single_clean": "single", "single_clean_rich": "single_rich"}
SEASONS = [("2425", "Test 2024/25, 202 matches", C["blue"], "o"), ("2526", "Held out 2025/26, 146 matches", C["orange"], "s")]


def main():
    R = Registry(NAME)
    hj = SRC["hold_json"]
    vals = {}   # (method index, season, metric) -> {seed: value}
    L = SRC["dest_rules"].read_text(encoding="utf8").split("\n")
    lr = next(i for i, x in enumerate(L) if x.startswith(f"| {RULE}")) + 1
    for k, (kind, d, model, label) in enumerate(METHODS):
        for met in ["top1", "skill"]:
            if kind == "rule":
                vals[(k, "2425", met)] = {0: R.md(f"rule/2425/{met}", SRC["dest_rules"], lr, met, fmt="f4")}
                vals[(k, "2526", met)] = {0: R.js(f"rule/2526/{met}", SRC["hold_dest_rules"], [RULE, met], fmt="f4")}
                continue
            if kind == "single":
                per = single_seed_values(d, "dest", model, "metric" if met == "top1" else "skill")
                ln = md_find(SRC["u1"], HEAD[d], 0, task="dest")
                ref = md_value(SRC["u1"], ln, model, "paren" if met == "top1" else "mean")
                key = f"{HKEY[d]}|{model}|dest"
                tag = f"{d}/{model}"
            else:
                per = unified_seed_values(d, "dest", "metric" if met == "top1" else "skill")
                ln = md_find(SRC["u1"], f"## Run {d} (", 0, task="dest")
                ref = md_value(SRC["u1"], ln, "test_mean" if met == "top1" else "dest_skill_mean")
                key = f"unified|{d}|dest"
                tag = d
            m = float(np.mean(list(per.values())))
            assert round(m, 4) == round(ref, 4), f"{tag} {met} 24/25 recomputed mean {m:.4f} differs from the summary table {ref}"
            if kind == "single":
                R.md(f"{tag}/2425/{met}/mean", SRC["u1"], ln, model, "paren" if met == "top1" else "mean", fmt="f4")
            else:
                R.md(f"{tag}/2425/{met}/mean", SRC["u1"], ln, "test_mean" if met == "top1" else "dest_skill_mean", fmt="f4")
            for s, v in per.items():
                R.perevent(f"{tag}/2425/{met}/s{s}", v, f"{tag} seed {s}, {met} on the 202 matches, recomputed from the per-event files", fmt="f4")
            vals[(k, "2425", met)] = per
            by = json_value(hj, ["metrics", key, met, "by_seed"])
            hm = json_value(hj, ["metrics", key, met, "mean"])
            assert abs(np.mean(list(by.values())) - hm) < 1e-12
            vals[(k, "2526", met)] = {int(s): R.js(f"{tag}/2526/{met}/s{s}", hj, ["metrics", key, met, "by_seed", s], fmt="f4") for s in by}
            R.js(f"{tag}/2526/{met}/mean", hj, ["metrics", key, met, "mean"], fmt="f4")

    rows = []
    for (k, se, met), per in vals.items():
        for s, v in per.items():
            rows.append(dict(method=METHODS[k][3], season=se, metric=met, seed=s, value=round(float(v), 6)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{NAME}_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    set_style_old()
    fig, axes = plt.subplots(1, 2, figsize=(17.5 * CM, 7.2 * CM), sharey=True)
    fig.subplots_adjust(left=0.24, right=0.985, bottom=0.14, top=0.9, wspace=0.07)
    y = np.arange(len(METHODS))[::-1].astype(float)
    cols = {"2425": OLD["navy"], "2526": OLD["red"]}
    bh = 0.36
    for ax, met, title, let in zip(axes, ["top1", "skill"], ["Top-1 accuracy", "Skill"], "ab"):
        ax.grid(axis="x", color=OLD["grid"], lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        rule25 = vals[(0, "2425", met)][0]
        ax.axvline(rule25, color="#5F5F5F", lw=0.9, ls=(0, (4, 2.5)), zorder=1)
        vmax = 0
        for k in range(len(METHODS)):
            for j, (se, slab, _, _) in enumerate(SEASONS):
                v = np.array(list(vals[(k, se, met)].values()))
                yy = y[k] + (bh / 2 + 0.01 if j == 0 else -bh / 2 - 0.01)
                m = v.mean()
                ax.barh(yy, m, height=bh, color=cols[se], edgecolor="white", linewidth=0.6, zorder=2)
                xe = m
                if len(v) > 1:
                    sd = v.std(ddof=1)
                    ax.errorbar(m, yy, xerr=sd, fmt="none", ecolor=OLD["ink"], elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=3)
                    xe = m + sd
                vlabel(ax, xe + 0.004 * (1 if met == "top1" else 1.6), yy, m, 4, va="center", ha="left", fontsize=7.5,
                        fontweight="bold", color=cols[se], zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.85))
                vmax = max(vmax, xe)
        ax.set_xlim(0, vmax * (1.5 if met == "top1" else 1.17))
        ax.set_ylim(-0.6, len(METHODS) - 0.4)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Share of passes with the true cell ranked first" if met == "top1" else "1 minus log loss over ln 96")
        panel_label(ax, let, title)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([m[3] for m in METHODS])
    from matplotlib.patches import Patch
    h = [Patch(facecolor=OLD["navy"], edgecolor="white", label="2024/25 test"),
         Patch(facecolor=OLD["red"], edgecolor="white", label="2025/26 held out"),
         Line2D([], [], color=OLD["ink"], lw=0.8, marker="|", ms=5, label="Seed SD"),
         Line2D([], [], color="#5F5F5F", lw=0.9, ls=(0, (4, 2.5)), label="Rule, 2024/25")]
    axes[0].legend(handles=h, loc="center right", bbox_to_anchor=(1.0, 0.5), borderaxespad=0.1, handlelength=1.5, labelspacing=0.3)
    save_fig(fig, NAME)
    R.save(extra=dict(data_csv=str(OUT_DIR / f"{NAME}_data.csv")))
    print(f"[{NAME}] done, {len(rows)} points")


if __name__ == "__main__":
    main()
