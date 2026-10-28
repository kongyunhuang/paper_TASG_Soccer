#!/usr/bin/env python3
"""
figS3_player_preference.py, D of the player passing preference for all players and for players who changed team
===============================================================================================================
Supplementary figure. D is the per-pass log loss of M3 minus M2.
  Left panel   per-player D (each player averaged over the three seeds within each fold, as in the --summarize step of
               player_decision_preference.py), as violins with the per-player points, a different marker for each fold,
               and the pooled estimate weighted by the number of test passes with its 95% bootstrap interval on top.
  Right panel  the y axis zoomed in around zero, with the pooled estimate, the estimate and interval of each fold, and the
               pooled point estimates of the three seeds.
The per-player values are recomputed from data/cache/player_pref/fold*_s*_players.parquet and the weighted mean is checked
against D pooled in results/player_decision_preference_summary.md. The intervals, the two folds and the seeds are taken
from the same summary table (the bootstrap intervals are copied, not resampled again).

Inputs   data/cache/player_pref/fold{1,2}_s{0,1,2}_players.parquet, results/player_decision_preference_summary.md
Outputs  outputs/figures/figS3_player_preference.{pdf,png}, results/figure_data/figS3_player_preference_{data,labels}.csv,
         outputs/provenance/figS3_player_preference_sources.json

Usage
  PYTHONPATH=. python -u scripts/paper/figS3_player_preference.py

Last modified 2026-10-28
"""
import csv
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scripts.paper.common import (CM, DEV, OLD, OUT_DIR, SRC, Registry, md_find, panel_label, ringed_scatter, save_fig, set_style_old,
                                  vlabel)

NAME = "figS3_player_preference"
P = SRC["pref"]
NUM = r"([-+]?[0-9.]+)"
GROUPS = [("All players", "All players", OLD["navy"], OLD["navy_l"]), ("Changed team", "Changed team", OLD["red"], OLD["red_l"])]


def per_player():
    files = sorted(glob.glob(str(DEV / "data" / "cache" / "player_pref" / "fold*_s*_players.parquet")))
    assert len(files) == 6, f"expected 6 files data/cache/player_pref/fold*_s*_players.parquet, found {len(files)} (run scripts/eval/player_decision_preference.py first)"
    allp = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    allp["d32"] = allp["M3"] - allp["M2"]
    avg = allp.groupby(["fold", "player_id"]).agg(d32=("d32", "mean"), n_test=("n_test", "first"), mover=("mover", "first"),
                                                  n_seeds=("seed", "nunique")).reset_index()
    assert (avg["n_seeds"] == 3).all()
    return avg


def main():
    R = Registry(NAME)
    avg = per_player()
    est = {}
    rows = []
    for grp, label, col, lcol in GROUPS:
        g = avg if grp == "All players" else avg[avg["mover"].astype(bool)]
        ln = md_find(P, "## Main quantity D", 0, **{"Group": grp})
        wm = float(np.average(g["d32"], weights=g["n_test"]))
        d = R.md(f"{grp}/D", P, ln, "D pooled", fmt="f5")
        assert round(wm, 5) == round(d, 5), f"{grp} weighted mean {wm:.5f} differs from the summary table {d}"
        R.perevent(f"{grp}/D_recomputed", wm, "M3 minus M2 in fold*_s*_players.parquet, averaged over the three seeds per player and fold, then averaged with n_test as weights", fmt="f5")
        n = R.md(f"{grp}/rows", P, ln, "Players (rows over both folds)", fmt="int")
        assert int(n) == len(g)
        R.perevent(f"{grp}/n_points", len(g), "number of per-player points (players times folds)", fmt="int")
        lo = R.mdre(f"{grp}/lo", P, ln, "95% CI", r"^\[" + NUM, fmt="sf5")
        hi = R.mdre(f"{grp}/hi", P, ln, "95% CI", r", " + NUM + r"\]$", fmt="sf5")
        f1 = R.mdre(f"{grp}/fold1", P, ln, "First fold", r"^" + NUM, fmt="sf5")
        f1lo = R.mdre(f"{grp}/fold1_lo", P, ln, "First fold", r"\[" + NUM, fmt="sf5")
        f1hi = R.mdre(f"{grp}/fold1_hi", P, ln, "First fold", r", " + NUM + r"\]", fmt="sf5")
        f2 = R.mdre(f"{grp}/fold2", P, ln, "Second fold", r"^" + NUM, fmt="sf5")
        f2lo = R.mdre(f"{grp}/fold2_lo", P, ln, "Second fold", r"\[" + NUM, fmt="sf5")
        f2hi = R.mdre(f"{grp}/fold2_hi", P, ln, "Second fold", r", " + NUM + r"\]", fmt="sf5")
        seeds = [R.mdre(f"{grp}/seed{k}", P, ln, "Pooled estimate per seed", r"^" + r"\S+\s+" * k + NUM, fmt="sf5") for k in range(3)]
        est[grp] = dict(d=d, lo=lo, hi=hi, folds=[(f1, f1lo, f1hi), (f2, f2lo, f2hi)], seeds=seeds, pts=g)
        for _, r in g.iterrows():
            rows.append(dict(group=label, fold=int(r["fold"]), player_id=int(r["player_id"]), d32=round(float(r["d32"]), 6), n_test=int(r["n_test"])))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{NAME}_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    set_style_old()
    fig, (ax, az) = plt.subplots(1, 2, figsize=(17.5 * CM, 7.0 * CM), gridspec_kw=dict(width_ratios=[1.15, 1]))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.2, top=0.88, wspace=0.25)
    rng = np.random.RandomState(11)
    mk = {1: "o", 2: "^"}
    for i, (grp, label, col, lcol) in enumerate(GROUPS):
        e = est[grp]
        g = e["pts"]
        vp = ax.violinplot([g["d32"].values], positions=[i], widths=0.7, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(OLD["band"])
            b.set_edgecolor("#BFBFBF")
            b.set_linewidth(0.6)
            b.set_alpha(1)
        for fo in (1, 2):
            gf = g[g["fold"] == fo]
            ax.scatter(i + (rng.rand(len(gf)) - 0.5) * 0.42, gf["d32"], s=5, marker=mk[fo], color=lcol, edgecolor="none", alpha=0.7, zorder=2)
        ax.errorbar(i, e["d"], yerr=[[e["d"] - e["lo"]], [e["hi"] - e["d"]]], fmt="none", ecolor=OLD["ink"], elinewidth=1.0, capsize=3, zorder=4)
        ringed_scatter(ax, [i], [e["d"]], col, s=34, marker="D")
    ax.axhline(0, color="#5F5F5F", lw=0.8, ls=(0, (4, 2.5)), zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{lab}\n({len(est[g]['pts'])} player-folds)" for g, lab, *_ in GROUPS])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("D, log loss of M3 minus M2")
    ax.tick_params(axis="x", length=0)
    panel_label(ax, "a", "Per-player D, both folds")
    h = [Line2D([], [], marker="o", ls="", color=OLD["gray"], ms=3.5, label="Player, fold 1"),
         Line2D([], [], marker="^", ls="", color=OLD["gray"], ms=3.8, label="Player, fold 2"),
         Line2D([], [], marker="D", ls="", color=OLD["ink"], ms=5, markeredgecolor="white", label="Pooled D, 95% CI")]
    ax.legend(handles=h, loc="upper right", borderaxespad=0.2, handletextpad=0.3, labelspacing=0.3)

    # right panel, zoomed in around zero
    az.grid(axis="y", color=OLD["grid"], lw=0.6, zorder=0)
    az.set_axisbelow(True)
    az.axhline(0, color="#5F5F5F", lw=0.8, ls=(0, (4, 2.5)), zorder=1)
    xpos = {"pooled": 0.0, "f1": 0.45, "f2": 0.9}
    for i, (grp, label, col, lcol) in enumerate(GROUPS):
        e = est[grp]
        base = i * 1.55
        az.errorbar(base, e["d"], yerr=[[e["d"] - e["lo"]], [e["hi"] - e["d"]]], fmt="none", ecolor=col, elinewidth=1.6, capsize=3.5, zorder=3)
        ringed_scatter(az, [base], [e["d"]], col, s=40, marker="D")
        vlabel(az, base, e["hi"] + 0.00025, e["d"], 5, ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=col, zorder=5)
        for k, (fv, flo, fhi) in enumerate(e["folds"]):
            x = base + (xpos["f1"] if k == 0 else xpos["f2"])
            az.errorbar(x, fv, yerr=[[fv - flo], [fhi - fv]], fmt="none", ecolor=lcol, elinewidth=1.2, capsize=2.5, zorder=3)
            az.scatter([x], [fv], s=22, marker=mk[k + 1], color=col, edgecolor="white", linewidth=0.5, zorder=4)
        for k, sv in enumerate(e["seeds"]):
            az.scatter([base - 0.2], [sv], s=14, marker="_", color=OLD["ink"], linewidth=1.0, zorder=5)
    az.set_xticks([0, 0.45, 0.9, 1.55, 2.0, 2.45])
    az.set_xticklabels(["Pooled", "Fold 1", "Fold 2", "Pooled", "Fold 1", "Fold 2"], rotation=0)
    for i, (grp, label, col, _) in enumerate(GROUPS):
        az.text(i * 1.55 + 0.45, -0.1, label, transform=az.get_xaxis_transform(), ha="center", va="top", fontsize=7.5, fontweight="bold", color=col)
    az.set_xlim(-0.45, 2.75)
    az.tick_params(axis="x", length=0)
    az.set_ylabel("D, zoomed")
    panel_label(az, "b", "Pooled, per-fold and per-seed estimates")
    hz = [Line2D([], [], marker="D", ls="", color=OLD["ink"], ms=5, markeredgecolor="white", label="Pooled, 95% CI"),
          Line2D([], [], marker="o", ls="", color=OLD["ink"], ms=4, label="Fold 1, 95% CI"),
          Line2D([], [], marker="^", ls="", color=OLD["ink"], ms=4.3, label="Fold 2, 95% CI"),
          Line2D([], [], marker="_", ls="", color=OLD["ink"], ms=6, markeredgewidth=1.0, label="Seeds 0, 1, 2, pooled")]
    az.legend(handles=hz, loc="upper left", borderaxespad=0.2, handletextpad=0.3, labelspacing=0.3)
    save_fig(fig, NAME)
    R.save(extra=dict(data_csv=str(OUT_DIR / f"{NAME}_data.csv")))
    print(f"[{NAME}] done, {len(rows)} per-player points")


if __name__ == "__main__":
    main()
