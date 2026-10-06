#!/usr/bin/env python3
"""
analyze_within_2425_shift: F4: residual within-24/25 drift diagnostic
======================================================================
After the relaxed holdout, the residual mean(p)-mean(y) is approximately
+0.04 to +0.10. We ask:
  (1) Is this bias uniform across 24/25 matches, or concentrated in a subset?
  (2) How does it differ between the EPL and La Liga?
  (3) Is the per-match sample size correlated with the bias?

Input:
  data/predictions/{task}_{model}_relaxed.npz  (from the rerun relaxed script)

Outputs:
  data/results_within_2425.json
  figures/within_2425_match_bias.png

Usage:
  PYTHONPATH=. python scripts/supplementary_plots/analyze_within_2425_shift.py

"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.plotting.plot_utils import set_paper_style

CACHE = Path("data")
PRED = CACHE / "predictions"
FIG = Path("figures"); FIG.mkdir(exist_ok=True)

COMBOS = [
    ("dribble", "M4_CNN_2ch"),
    ("dribble", "G1_Gating_2ch"),
    ("duel", "M4_CNN_2ch"),
    ("duel", "G1_Gating_2ch"),
]


def league_of(season_dir):
    """season_dir like '2_317_2024_25' or '11_317_2024_25'."""
    try:
        comp_id = season_dir.split("_")[0]
        return "EPL" if comp_id == "2" else ("La Liga" if comp_id == "11" else f"comp{comp_id}")
    except Exception:
        return "?"


def per_match_stats(y, p, match_ids, min_n=10):
    ids = np.unique(match_ids)
    rows = []
    for mid in ids:
        m = match_ids == mid
        n = int(m.sum())
        if n < min_n:
            continue
        rows.append({
            "match_id": str(mid),
            "n": n,
            "mean_y": float(y[m].mean()),
            "mean_p": float(p[m].mean()),
            "bias": float(p[m].mean() - y[m].mean()),
        })
    return rows


def main():
    set_paper_style()
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    all_reports = []
    # Layout: 2 rows (top = bias histogram, bottom = n×bias scatter) × 4 cols (combos)
    fig, axes = plt.subplots(2, len(COMBOS), figsize=(16, 7), dpi=150,
                             gridspec_kw={"hspace": 0.42, "wspace": 0.32})

    for col, (task, model) in enumerate(COMBOS):
        path = PRED / f"{task}_{model}_relaxed.npz"
        if not path.exists():
            print(f"[skip] {path} not found yet")
            continue
        d = np.load(path, allow_pickle=True)
        y = d["y_test"].astype(float)
        p = d["probs"].astype(float)
        mid = d["match_id"]
        season = d["season_dir"]
        league = np.array([league_of(s) for s in season])

        rows = per_match_stats(y, p, mid, min_n=5)
        biases = np.array([r["bias"] for r in rows])
        # By league
        by_league = {}
        for L in ("EPL", "La Liga"):
            mask = league == L
            if mask.sum() == 0:
                continue
            by_league[L] = {
                "n_events": int(mask.sum()),
                "mean_y": float(y[mask].mean()),
                "mean_p": float(p[mask].mean()),
                "bias": float(p[mask].mean() - y[mask].mean()),
            }

        report = {
            "task": task,
            "model": model,
            "overall": {
                "n_events": int(len(y)),
                "n_matches": int(len({*mid})),
                "mean_y": float(y.mean()),
                "mean_p": float(p.mean()),
                "bias": float(p.mean() - y.mean()),
            },
            "by_league": by_league,
            "match_bias": {
                "median": float(np.median(biases)),
                "mean": float(biases.mean()),
                "std": float(biases.std()),
                "pct_biased_pos_gt_0.05": float((biases > 0.05).mean()),
                "pct_biased_neg_lt_-0.05": float((biases < -0.05).mean()),
                "pct_unbiased_abs_le_0.05": float((np.abs(biases) <= 0.05).mean()),
            },
        }
        all_reports.append(report)

        combo_label = f"{task} × {model.replace('_2ch','')}"

        # Top row panel: histogram of per-match bias
        # (overall bias annotated inline above the red line to avoid
        # legend boxes covering hist bars; league legend moved to fig level)
        ax = axes[0, col]
        ax.hist(biases, bins=22, color="#A0A0A0", edgecolor="black", linewidth=0.4)
        ax.axvline(0, color="#333", linestyle="--", linewidth=0.8)
        bias_val = report["overall"]["bias"]
        ax.axvline(bias_val, color="#CC3311", linewidth=1.4)
        # inline annotation above the red line (top of axes), small white pad
        ymax_hist = ax.get_ylim()[1]
        ax.text(bias_val, ymax_hist * 0.97,
                f"overall = {bias_val:+.3f}",
                color="#CC3311", fontsize=8.5, va="top", ha="center",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=1.5),
                zorder=6)
        ax.set_title(combo_label, fontsize=11.5, loc="left", pad=4)
        ax.set_xlabel("per-match bias  (mean_p − mean_y)", fontsize=10)
        if col == 0:
            ax.set_ylabel("# matches", fontsize=10.5)
        ax.grid(alpha=0.25, linestyle=":")
        ax.tick_params(labelsize=10)

        # Bottom row panel: scatter n vs bias, colored by league
        # (per-panel league legend dropped; shared fig.legend at top of figure)
        ax2 = axes[1, col]
        row_league = []
        for r in rows:
            rid = r["match_id"]
            sel = (mid.astype(str) == rid)
            row_league.append(league_of(season[sel][0]) if sel.any() else "?")
        for L, color in [("EPL", "#1F4E79"), ("La Liga", "#EE7733")]:
            xs = [r["n"] for r, lg in zip(rows, row_league) if lg == L]
            ys = [r["bias"] for r, lg in zip(rows, row_league) if lg == L]
            if xs:
                ax2.scatter(xs, ys, s=28, alpha=0.6, c=color,
                            label=f"{L} (n={len(xs)})" if col == 0 else None,
                            edgecolors="black", linewidths=0.3)
        ax2.axhline(0, color="#333", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("events per match", fontsize=10.5)
        if col == 0:
            ax2.set_ylabel("per-match bias", fontsize=10.5)
        ax2.grid(alpha=0.25, linestyle=":")
        ax2.tick_params(labelsize=10)

    fig.suptitle("Within-24/25 residual bias analysis  (after relaxed holdout)\n"
                 "rows: per-match bias histogram (top), bias vs match size by league (bottom)",
                 fontsize=12.5, y=1.04)
    # Shared figure-level league legend (EPL blue / La Liga orange) below suptitle
    from matplotlib.lines import Line2D
    league_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#1F4E79",
               markeredgecolor="black", markersize=8, label="EPL"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#EE7733",
               markeredgecolor="black", markersize=8, label="La Liga"),
    ]
    fig.legend(handles=league_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=10,
               frameon=False)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    png = FIG / "within_2425_match_bias.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "within_2425_match_bias.pdf", bbox_inches="tight")
    print(f"Wrote {png}")

    out = CACHE / "results_within_2425.json"
    with open(out, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"Wrote {out}")

    # Print summary
    print("\n" + "=" * 78)
    print(f"{'task':<10}{'model':<15}{'overall bias':>14}{'EPL bias':>12}{'LaLiga bias':>12}{'% biased':>12}")
    print("-" * 78)
    for r in all_reports:
        epl_b = r["by_league"].get("EPL", {}).get("bias", float("nan"))
        ll_b = r["by_league"].get("La Liga", {}).get("bias", float("nan"))
        pct_high = r["match_bias"]["pct_biased_pos_gt_0.05"]
        print(f"{r['task']:<10}{r['model']:<15}"
              f"{r['overall']['bias']:>+14.4f}"
              f"{epl_b:>+12.4f}{ll_b:>+12.4f}"
              f"{pct_high*100:>10.1f}%")


if __name__ == "__main__":
    main()
