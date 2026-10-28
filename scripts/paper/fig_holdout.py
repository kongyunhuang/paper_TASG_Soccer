#!/usr/bin/env python3
"""
fig_holdout.py, quantities of the four held-out claims in 2024/25 and 2025/26
=============================================================================
For each claim the quantities of 2024/25 against 2025/26, with a 95% match bootstrap interval for 2025/26 and the
threshold to confirm marked on every row.
The quantities, intervals and 2024/25 references of C1, C2 and C3 come from results/holdout_2526_summary.md (the same
rows as tab_holdout), the thresholds from the threshold dictionary at the top of the same file, and the verdict words
from the recorded verdicts of results/decision_rules.md. The 2025/26 per-seed gates of C4 come from derived.C4 of
results/holdout_2526_summary.json; the 2024/25 per-seed gates are recomputed from the u1_clean per-event files (their
means checked against results/u1_clean_summary.md). The gate readings have no bootstrap interval.

Inputs   results/holdout_2526_summary.md, results/holdout_2526_summary.json, results/decision_rules.md,
         data/cache/u1_clean/{u1,shuf_dest}_s*.npz, results/u1_clean_summary.md
Outputs  outputs/figures/fig_holdout.{pdf,png}, results/figure_data/fig_holdout_{data,labels}.csv,
         outputs/provenance/fig_holdout_sources.json

Usage
  PYTHONPATH=. python -u scripts/paper/fig_holdout.py

Last modified 2026-10-28
"""
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.paper.common import (C, CM, OLD, OUT_DIR, SRC, Registry, json_value, md_find, md_row, md_value, rel, ringed_scatter,
                                  save_fig, set_style_old, unified_seed_values, vlabel)

NAME = "fig_holdout"
H, J, U, F = SRC["hold_md"], SRC["hold_json"], SRC["u1"], SRC["decision_rules"]


def num(v, nd):
    """Number for a label, with a math minus sign"""
    return f"{v:.{nd}f}".replace("-", "−")


def thr_value(R, key, iid):
    L = H.read_text(encoding="utf8").split("\n")
    ln = next(i for i, x in enumerate(L) if f'"{key}"' in x) + 1
    v = float(re.search(r":\s*([-0-9.]+)", L[ln - 1]).group(1))
    R.add(iid, v, dict(kind="mdline", file=rel(H), line=ln, regex=r":\s*([-0-9.]+)"), fmt="f3")
    return v


def main():
    R = Registry(NAME)
    panels = []   # (title, list of rows, x-axis label); a row is dict(label, v25, v26, lo, hi, thr, kind)

    # C1
    rows = []
    for label, iid, key, txt in [("u1fcn minus strongest rule", "C1/fcn_minus_rule", "fcn_minus_rule_confirm", "Variant minus rule\n"),
                                 ("u1fcn minus M2 enriched", "C1/fcn_minus_m2rich", "fcn_minus_m2rich_confirm", "Variant minus\nenriched MLP ")]:
        ln = md_find(H, "## C1", 0, **{"Quantity": label})
        t = thr_value(R, key, iid + "/thr")
        rows.append(dict(label=f"{txt}(≥ {num(t, 3)})", v25=R.md(iid + "/2425", H, ln, "24/25"), v26=R.md(iid + "/2526", H, ln, "Held-out"),
                         lo=R.md(iid + "/lo", H, ln, "95% CI", "lo"), hi=R.md(iid + "/hi", H, ln, "95% CI", "hi"), thr=[t]))
    panels.append(("C1", "Pass destination, spatial structure cannot be replaced", rows, "Difference in top-1 accuracy"))

    # C2
    rows = []
    for md_label, col, iid, key, txt, op in [
            ("XGB enriched minus M4 thin, held-out", "Three-task mean", "C2/a", "xgbrich_minus_m4thin_confirm", "Enriched XGBoost\nminus thin CNN", "≥"),
            ("Enriched gain M4 minus M2, held-out", "Three-task mean", "C2/b", "rich_gain_abs_confirm", "Enriched gain\n", "≤"),
            ("Thin gain M4 minus M2, held-out", "Three-task mean", "C2/thin", None, "Gain, thin scalars\n(reference for enriched gain)", None),
            ("Thin gain M4 minus M2, held-out", "pressure", "C2/c_thin", "pressure_thin_gain_min", "Pressure gain,\nthin", "≥"),
            ("Enriched gain M4 minus M2, held-out", "pressure", "C2/c_rich", "pressure_rich_gain_max", "Pressure enriched\ngain", "≤")]:
        ln = md_find(H, "## C2", 0, **{"Quantity": md_label})
        assert md_row(H, ln + 1)["Quantity"] == "Same, 95% CI" and md_row(H, ln + 2)["Quantity"] == "Same, 24/25"
        t = thr_value(R, key, iid + "/thr") if key else None
        lab = f"{txt} ({op} {num(t, 3)})" if key else txt
        rows.append(dict(label=lab, v25=R.md(iid + "/2425", H, ln + 2, col), v26=R.md(iid + "/2526", H, ln, col),
                         lo=R.md(iid + "/lo", H, ln + 1, col, "lo"), hi=R.md(iid + "/hi", H, ln + 1, col, "hi"), thr=[t] if key else []))
    panels.append(("C2", "Outcome tasks, enriched scalars replace most of the spatial gain", rows, "Difference in AUC, mean over Pass, Ball recovery and Pressure unless Pressure is named"))

    # C3
    rows = []
    ln = md_find(H, "## C3", 0, **{"Quantity": "u1 minus XGB enriched, held-out"})
    for col, iid, key, txt in [("pass", "C3/pass", "pass_abs_confirm", "Pass"), ("pressure", "C3/pressure", "pressure_abs_confirm", "Pressure"),
                               ("ball_recovery", "C3/br", "br_abs_confirm", "Ball recovery\n"),
                               ("Three-task mean (Pass BR Pressure)", "C3/mean3", "mean3_abs_confirm", "Mean of three\n")]:
        t = thr_value(R, key, iid + "/thr")
        rows.append(dict(label=f"{txt}{'' if txt.endswith(chr(10)) else ' '}(|d| ≤ {num(t, 3)})", v25=R.md(iid + "/2425", H, ln + 2, col), v26=R.md(iid + "/2526", H, ln, col),
                         lo=R.md(iid + "/lo", H, ln + 1, col, "lo"), hi=R.md(iid + "/hi", H, ln + 1, col, "hi"), thr=[-t, t], band=True))
    panels.append(("C3", "Unified model stays close to enriched XGBoost", rows, "Unified model minus enriched XGBoost, AUC"))

    # C4
    rows = []
    for tag, key, txt, op in [("u1", "u1_dest_gate_min", "Unified model\n", "≥"),
                              ("shuf_dest", "shuf_dest_gate_max", "Destination shuffled\n", "≤")]:
        per25 = unified_seed_values(tag, "dest", "gate")
        lu = md_find(U, f"## Run {tag} (", 0, task="dest")
        assert round(float(np.mean(list(per25.values()))), 4) == round(md_value(U, lu, "gate_mean"), 4)
        R.md(f"C4/{tag}/2425_mean", U, lu, "gate_mean", fmt="f4")
        for s, v in per25.items():
            R.perevent(f"C4/{tag}/2425/s{s}", v, f"mean gate_mean of {tag}_s{s}.npz over the destination events, 202 matches", fmt="f4")
        seeds = sorted(json_value(J, ["derived", "C4", tag]).keys())
        per26 = {int(s): R.js(f"C4/{tag}/2526/s{s}", J, ["derived", "C4", tag, s, "dest_gate"], fmt="f4") for s in seeds}
        t = thr_value(R, key, f"C4/{tag}/thr")
        m25 = next(it["value"] for it in R.items if it["id"] == f"C4/{tag}/2425_mean")
        m26 = R.derived(f"C4/{tag}/2526_mean", float(np.mean(list(per26.values()))), "mean of these seeds", [f"C4/{tag}/2526/s{s}" for s in seeds], fmt="f3")
        rows.append(dict(label=f"{txt}({op} {num(t, 2)})", s25=list(per25.values()), s26=list(per26.values()), thr=[t], m25=m25, m26=m26))
    panels.append(("C4", "Destination gate pattern on new matches, gate reading only", rows, "Mean gate value of the destination task, one point per seed"))

    # verdict words
    FL = F.read_text(encoding="utf8").split("\n")
    v0 = next(i for i, x in enumerate(FL) if x.startswith("## Recorded verdicts"))
    words = {"C1": ("confirmed", "Confirmed"), "C2": ("confirmed, Pressure near the bound", "Confirmed, Pressure near the bound"), "C3": ("inconclusive", "Inconclusive"), "C4": ("holds", "Holds")}
    verdict = {}
    for c, (w, must) in words.items():
        ln = next(i for i, x in enumerate(FL) if i > v0 and x.startswith(f"| holdout_{c} ")) + 1
        verdict[c] = R.text(f"{c}/verdict", w, F, ln, must)

    # table view
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
    # left column C1 and C3, right column C2 and C4, the panel heights of each column follow their numbers of rows
    gl = GridSpec(2, 1, figure=fig, left=0.19, right=0.43, top=0.92, bottom=0.1, hspace=0.62,
                  height_ratios=[nrow["C1"] + 0.9, nrow["C3"] + 0.9])
    gr = GridSpec(2, 1, figure=fig, left=0.705, right=0.935, top=0.92, bottom=0.1, hspace=0.62,
                  height_ratios=[nrow["C2"] + 0.9, nrow["C4"] + 0.9])
    axd = {"C1": fig.add_subplot(gl[0]), "C3": fig.add_subplot(gl[1]), "C2": fig.add_subplot(gr[0]), "C4": fig.add_subplot(gr[1])}
    short = {"C1": "C1, pass destination", "C2": "C2, outcome tasks", "C3": "C3, unified model", "C4": "C4, destination gate"}
    xlabs = {"C1": "Difference in top-1 accuracy", "C2": "AUC difference, mean of Pass,\nBall recovery, Pressure unless named",
             "C3": "Unified minus enriched XGBoost, AUC", "C4": "Destination gate value, points are seeds"}
    fmt = lambda v, nd: f"{v:.{nd}f}".replace("-", "−")
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
    print(f"[{NAME}] done, {len(out)} points")


if __name__ == "__main__":
    main()
