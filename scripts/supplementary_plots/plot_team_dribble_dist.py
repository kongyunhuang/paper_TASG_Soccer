#!/usr/bin/env python3
"""
plot_team_dribble_dist: Case Study 5: 24/25 team-level dribble-quality distribution
====================================================================================
Uses the dribble M4 CNN (2-channel) relaxed-holdout predictions, joins them
with L1 events to recover team_name, computes the per-team dribble-quality
distribution (per-event quality = y_true - y_pred), then draws a box plot
sorted by median quality, highlighting the top 5 and bottom 5.

Inputs:
  data/predictions/dribble_M4_CNN_2ch_relaxed.npz
  data/L1_events_v3.parquet  (event_id -> team_name)
Outputs:
  data/results_team_dribble_quality.json
  figures/figS5.png + .pdf

Usage:
  PYTHONPATH=. python scripts/supplementary_plots/plot_team_dribble_dist.py

"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plotting.plot_utils import set_paper_style

CACHE_DIR = Path("data")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True, parents=True)

MIN_N_PER_TEAM = 30   # filter out teams with too few samples
TOP_N = 5
BOTTOM_N = 5


def main():
    set_paper_style()
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    print("Loading dribble M4 relaxed predictions ...")
    d = np.load(CACHE_DIR / "predictions/dribble_M4_CNN_2ch_relaxed.npz",
                allow_pickle=True)
    eids = d["event_id"]
    y = d["y_test"].astype(np.float32)
    p = d["probs"].astype(np.float32)
    quality = y - p

    print(f"  n_events={len(eids):,}  mean_q={quality.mean():+.4f}")

    print("Loading L1 events for team_name join ...")
    cols = ["event_id", "team_name", "season_dir", "type_name"]
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet", columns=cols)

    df_e = pd.DataFrame({"event_id": eids, "y": y, "p": p, "quality": quality})
    df_m = df_e.merge(df[["event_id", "team_name"]], on="event_id", how="left")
    n_missing = df_m["team_name"].isna().sum()
    if n_missing > 0:
        print(f"  WARN: {n_missing} events missing team_name (dropped)")
    df_m = df_m.dropna(subset=["team_name"]).reset_index(drop=True)

    grp = df_m.groupby("team_name")
    summary = grp["quality"].agg(["count", "mean", "median", "std"]).reset_index()
    summary = summary.rename(columns={"count": "n", "mean": "mean_q",
                                       "median": "median_q", "std": "std_q"})
    summary = summary[summary["n"] >= MIN_N_PER_TEAM].copy()
    summary = summary.sort_values("median_q", ascending=True).reset_index(drop=True)
    print(f"  Teams ≥{MIN_N_PER_TEAM} dribbles: {len(summary)}")

    out_json = {
        "n_total_events": int(len(df_m)),
        "min_n_per_team": MIN_N_PER_TEAM,
        "global_mean_q": float(df_m["quality"].mean()),
        "global_median_q": float(df_m["quality"].median()),
        "teams": summary.to_dict(orient="records"),
    }
    json_path = CACHE_DIR / "results_team_dribble_quality.json"
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"  Saved {json_path}")

    print("\nBottom 5 (low quality):")
    for _, row in summary.head(5).iterrows():
        print(f"  {row['team_name']:30s} n={int(row['n']):4d} median={row['median_q']:+.3f} mean={row['mean_q']:+.3f}")
    print("Top 5 (high quality):")
    for _, row in summary.tail(5).iterrows():
        print(f"  {row['team_name']:30s} n={int(row['n']):4d} median={row['median_q']:+.3f} mean={row['mean_q']:+.3f}")

    # -- Draw a horizontal box plot (used in place of a violin: dribble quality
    #    is a binary-derived quantity, so a violin's twin humps mask median
    #    differences; the box plot highlights median, IQR and spread instead). --
    print("\nPlotting horizontal box plot ...")
    teams = summary["team_name"].tolist()
    quality_per_team = [df_m.loc[df_m["team_name"] == t, "quality"].values for t in teams]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.22 * len(teams))))
    pos = np.arange(len(teams))

    bp = ax.boxplot(quality_per_team, positions=pos, vert=False, widths=0.62,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.6, zorder=5),
                    whiskerprops=dict(color="#444", linewidth=0.9),
                    capprops=dict(color="#444", linewidth=0.9),
                    boxprops=dict(linewidth=0.7))

    # Colours: top N Tol blue / bottom N Tol red / others light grey (matches the main paper palette)
    for i, box in enumerate(bp["boxes"]):
        if i < BOTTOM_N:
            box.set_facecolor("#CC3311"); box.set_edgecolor("#7f1c14"); box.set_alpha(0.78)
        elif i >= len(teams) - TOP_N:
            box.set_facecolor("#1F4E79"); box.set_edgecolor("#0F2C4A"); box.set_alpha(0.78)
        else:
            box.set_facecolor("#A0A0A0"); box.set_edgecolor("#606060"); box.set_alpha(0.62)

    # Overlay mean markers (red/green/grey dots) to expose any right-tail bias where mean differs from median
    means = [np.mean(q) for q in quality_per_team]
    for i, m in enumerate(means):
        ax.plot([m], [pos[i]], marker="D", markersize=4.0,
                markerfacecolor="white", markeredgecolor="black",
                markeredgewidth=0.8, zorder=6)

    # Global-median dashed line
    gmed = float(df_m["quality"].median())
    ax.axvline(gmed, color="#5F5F5F", linewidth=1.0, linestyle="--",
               alpha=0.7, zorder=2,
               label=f"League median ({gmed:+.3f})")
    ax.axvline(0.0, color="#888", linewidth=0.7, linestyle=":",
               alpha=0.5, zorder=1)

    # legend with mean diamond
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#5F5F5F", linewidth=1.0, linestyle="--",
               label=f"League median ({gmed:+.3f})"),
        Line2D([0], [0], color="black", linewidth=1.6, label="Per-team median"),
        Line2D([0], [0], marker="D", markerfacecolor="white", markeredgecolor="black",
               linestyle="", markersize=4.5, label="Per-team mean (diamond = right-tail bias)"),
    ]
    # Legend in upper-right corner inside axes (xlim -0.7..+0.7).
    # n=XXX team-size labels sit at x=1.06 OUTSIDE axes, so loc="upper right"
    # never touches them.
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8.0,
              framealpha=0.92, bbox_to_anchor=(1.20, 1.0))

    ax.set_yticks(pos)
    ax.set_yticklabels([t if len(t) <= 22 else t[:21] + "…" for t in teams], fontsize=8.5)
    ax.set_xlabel("Dribble quality  =  actual success − model-predicted success",
                  fontsize=10)
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, len(teams) - 0.3)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.set_title(
        f"Per-team dribble quality (24/25 holdout, n={int(summary['n'].sum())} events, "
        f"{len(teams)} teams ≥ {MIN_N_PER_TEAM} dribbles)\n"
        f"Box = IQR (Q1–Q3); whiskers = 1.5×IQR clipped to ±1; black bar = median; diamond = mean",
        fontsize=10.5, pad=10)

    # Annotate per-team n
    for i, n in enumerate(summary["n"].values):
        ax.text(1.06, pos[i], f"n={int(n)}", fontsize=7.5, va="center", ha="left",
                color="#555")

    fig.tight_layout()
    out_png = FIG_DIR / "figS5.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figS5.pdf", bbox_inches="tight")
    print(f"  Saved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
