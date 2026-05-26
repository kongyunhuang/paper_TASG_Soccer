#!/usr/bin/env python3
"""
plot_player_case_study: player case study (horizontal 3-column, attacking-half zoom)
======================================================================================
For each of the top-3 dribblers, picks a representative dribble and draws the
freeze-frame, ball, predicted P and observed outcome on a pitch zoomed to the
attacking half (x in [60, 120]).

Change log (2026-04-26 redraw):
  - 3 vertical rows -> 3 horizontal columns, making full use of the page width
  - xlim restricted to [58, 122]: all players are in the attacking half, so the
    left half is cropped away
  - annotations moved above the pitch into the caption strip, no longer covering
    the freeze-frame
  - larger font sizes (player name 14pt, annotation 10pt)

Inputs:
  data/predictions/dribble_M4_CNN_2ch_relaxed.npz
  data/results_dribble_ranking.json
  data/L1_events_v3.parquet
  data/action_soccermaps_dribble.npy + _idx.parquet

Output:
  figures/figS2.png / .pdf

Usage:
  PYTHONPATH=. python scripts/supplementary_plots/plot_player_case_study.py

"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plotting.plot_utils import (COLOR_BALL, COLOR_OPPONENT, COLOR_TEAMMATE,
                                 draw_pitch, set_paper_style)

CACHE = Path("data")
PRED = CACHE / "predictions"
FIG = Path("figures"); FIG.mkdir(exist_ok=True)


def scatter_players_from_grid(ax, smap_2ch, jitter_seed=0, x_min_yards=58):
    """
    Convert (2, 8, 12) team-distribution grid to scattered player markers.
    Skip cells whose centre is left of x_min_yards (we only show attacking half).
    """
    rng = np.random.default_rng(jitter_seed)
    cell_w = 120 / 12
    cell_h = 80 / 8
    for ch, color in [(0, COLOR_TEAMMATE), (1, COLOR_OPPONENT)]:
        for gy in range(8):
            for gx in range(12):
                count = int(round(float(smap_2ch[ch, gy, gx])))
                if count <= 0:
                    continue
                cx = (gx + 0.5) * cell_w
                cy = (gy + 0.5) * cell_h
                if cx < x_min_yards:
                    continue
                for _ in range(count):
                    x = cx + (rng.random() - 0.5) * cell_w * 0.7
                    y = cy + (rng.random() - 0.5) * cell_h * 0.7
                    ax.scatter([x], [y], s=170, c=color, marker="o",
                               edgecolors="black", linewidths=0.7,
                               alpha=0.92, zorder=3)


def main():
    set_paper_style()
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    print("Loading rankings ...")
    with open(CACHE / "results_dribble_ranking.json") as f:
        rankings = json.load(f)
    top3 = rankings["M4_CNN_2ch"]["top20"][:3]

    print("Loading predictions + events ...")
    d = np.load(PRED / "dribble_M4_CNN_2ch_relaxed.npz", allow_pickle=True)
    df_pred = pd.DataFrame({
        "event_id": d["event_id"].astype(str),
        "y_true": d["y_test"].astype(float),
        "p": d["probs"].astype(float),
    })
    df_ev = pd.read_parquet(CACHE / "L1_events_v3.parquet",
                            columns=["event_id", "player_id", "player_name",
                                     "team_name", "location_x", "location_y",
                                     "season_dir"])
    df_ev["event_id"] = df_ev["event_id"].astype(str)
    df = df_pred.merge(df_ev, on="event_id", how="left")

    idx_df = pd.read_parquet(CACHE / "action_soccermaps_dribble_idx.parquet")
    smaps = np.load(CACHE / "action_soccermaps_dribble.npy", mmap_mode="r")
    smap_lookup = {str(eid): i for i, eid in enumerate(idx_df["event_id"].values)}

    selected = []
    for player_info in top3:
        player_id = int(float(player_info["player_id"]))
        player_name = player_info["player_name"]
        team = player_info["team_name"]
        sub = df[df["player_id"] == player_id]
        succ = sub[sub["y_true"] == 1]
        if len(succ) == 0:
            continue
        # Prefer attacking-half (x > 80) successes for visual zoom; fall back gracefully
        atk = succ[succ["location_x"] > 80]
        pool = atk if len(atk) else succ[succ["location_x"] > 60]
        if len(pool) == 0:
            pool = succ
        best = pool.sort_values("p", ascending=False).iloc[0]
        selected.append({
            "player_name": player_name, "team": team,
            "event_id": best["event_id"],
            "y_true": int(best["y_true"]), "p": float(best["p"]),
            "loc_x": float(best["location_x"]), "loc_y": float(best["location_y"]),
            "n_dribbles_in_test": int(player_info["n"]),
            "season_quality": float(player_info["quality"]),
        })

    n_panels = len(selected)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.6 * n_panels, 6.0),
                             dpi=150, gridspec_kw={"wspace": 0.10})
    if n_panels == 1:
        axes = [axes]

    XMIN_PLOT = 58.0  # zoom: attacking half + small left margin
    for idx, (ax, s) in enumerate(zip(axes, selected)):
        eid = str(s["event_id"])
        if eid not in smap_lookup:
            continue
        smap = np.asarray(smaps[smap_lookup[eid]][:2, :, :])  # (2, 8, 12)

        draw_pitch(ax)
        # Override pitch xlim to zoom to attacking half
        ax.set_xlim(XMIN_PLOT, 122)
        ax.set_ylim(-3, 96)  # Add headroom above 80 for annotation banner

        scatter_players_from_grid(ax, smap, jitter_seed=idx,
                                  x_min_yards=XMIN_PLOT)

        # Ball at recorded location
        ax.scatter([s["loc_x"]], [s["loc_y"]], c=COLOR_BALL, marker="D",
                   s=260, edgecolors="black", linewidths=1.4, zorder=5)

        # Banner area above pitch (y ∈ [83, 95])
        # Player name (large)
        ax.text(90, 93, s["player_name"], fontsize=14, fontweight="bold",
                ha="center", va="center", color="#1a1a1a")
        ax.text(90, 88,
                f"{s['team']}   |   season quality {s['season_quality']:+.3f}   "
                f"(n={s['n_dribbles_in_test']} in 24/25 holdout)",
                fontsize=10, ha="center", va="center", color="#555",
                style="italic")

        # Bottom annotation strip (y ∈ [-2.5, -2.5]): rendered as inset under pitch
        outcome = "SUCCESS" if s["y_true"] == 1 else "FAIL"
        outcome_col = "#1F4E79" if s["y_true"] == 1 else "#CC3311"
        ax.text(90, -8, f"This dribble:  P(success) = {s['p']:.2f}",
                fontsize=11, ha="center", va="center", color="#222")
        ax.text(90, -12.5, f"observed outcome:  {outcome}",
                fontsize=10.5, ha="center", va="center",
                color=outcome_col, fontweight="bold")
        ax.set_ylim(-15, 96)

    fig.suptitle("Top-3 dribblers by quality residual :  signature successful dribble",
                 fontsize=13.5, y=1.02)

    # Single legend at bottom of figure
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOR_TEAMMATE,
               markeredgecolor="black", markersize=10, label="Teammate"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOR_OPPONENT,
               markeredgecolor="black", markersize=10, label="Opponent"),
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor=COLOR_BALL,
               markeredgecolor="black", markersize=10, label="Ball (dribble start)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])

    png = FIG / "figS2.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "figS2.pdf", bbox_inches="tight")
    print(f"Wrote {png}")
    for s in selected:
        print(f"  {s['player_name']:>22s} ({s['team']:>14s})  "
              f"P={s['p']:.3f}  loc=({s['loc_x']:.1f}, {s['loc_y']:.1f})  "
              f"quality={s['season_quality']:+.3f}")


if __name__ == "__main__":
    main()
