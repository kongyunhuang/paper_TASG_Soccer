#!/usr/bin/env python3
"""
plot_method_overview: Figure 1 candidate (architecture schematic / paper-iconic)
==============================================================
Shows how a single Dribble event flows through our architecture:
  (a) Raw event + freeze-frame on the pitch.
  (b) SoccerMap 7-channel encoding (small grid montage showing each channel).
  (c) Gating mechanism + output P(success).

Inputs:
  data/predictions/dribble_M4_CNN_2ch_relaxed.npz
  data/L1_events_v3.parquet (with dribble x/y)
  data/action_soccermaps_dribble.npy + _idx.parquet

Outputs:
  figures/fig1.png / .pdf

Usage:
  PYTHONPATH=. python scripts/plotting/plot_method_overview.py

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scripts.plotting.plot_utils import (COLOR_BALL, COLOR_OPPONENT, COLOR_TEAMMATE,
                                 draw_pitch, set_paper_style)

CACHE = Path("data")
PRED = CACHE / "predictions"
FIG = Path("figures"); FIG.mkdir(exist_ok=True)

CHANNEL_LABELS = ["Ch0\nTeammates", "Ch1\nOpponents", "Ch2\nDist → ball",
                  "Ch3\nDist → goal", "Ch4\nsin ∠ ball",
                  "Ch5\ncos ∠ ball", "Ch6\n∠ to goal"]


def main():
    set_paper_style()
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    print("Loading ...")
    d = np.load(PRED / "dribble_M4_CNN_2ch_relaxed.npz", allow_pickle=True)
    df_pred = pd.DataFrame({
        "event_id": d["event_id"].astype(str),
        "y_true": d["y_test"].astype(float),
        "p": d["probs"].astype(float),
    })
    df_ev = pd.read_parquet(CACHE / "L1_events_v3.parquet",
                            columns=["event_id", "player_name", "team_name",
                                     "location_x", "location_y"])
    df_ev["event_id"] = df_ev["event_id"].astype(str)
    df = df_pred.merge(df_ev, on="event_id", how="left")

    # Pick a "perfect example": confident success in attacking third
    cands = df[(df["y_true"] == 1) & (df["p"] > 0.85) &
               (df["location_x"] > 80)].copy()
    cands = cands.sort_values("p", ascending=False)
    if len(cands) == 0:
        print("No matching example, falling back")
        cands = df[df["y_true"] == 1].sort_values("p", ascending=False)
    chosen = cands.iloc[0]
    print(f"Chosen event_id={chosen['event_id']}  player={chosen['player_name']}  "
          f"P={chosen['p']:.3f}  loc=({chosen['location_x']:.1f}, {chosen['location_y']:.1f})")

    # Get SoccerMap input (full 7 channels: the underlying npy is 7ch)
    idx_df = pd.read_parquet(CACHE / "action_soccermaps_dribble_idx.parquet")
    smaps = np.load(CACHE / "action_soccermaps_dribble.npy", mmap_mode="r")
    id_map = {str(eid): i for i, eid in enumerate(idx_df["event_id"].values)}
    eid = str(chosen["event_id"])
    if eid not in id_map:
        print("Event not in SoccerMap cache!")
        return
    smap_full = np.asarray(smaps[id_map[eid]])  # (7, 8, 12)
    bx, by = float(chosen["location_x"]), float(chosen["location_y"])

    # Layout: 3 columns
    #   (a) wide: pitch with team distribution (1.6×)
    #   (b) middle: 7-channel grid (compact)
    #   (c) right: gating + output schematic
    fig = plt.figure(figsize=(17, 5.6), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.7, 1.3], wspace=0.20)

    # === (a) Pitch view ===
    ax_a = fig.add_subplot(gs[0, 0])
    draw_pitch(ax_a)
    # Reconstruct approximate player dots from teammate / opponent grids
    cell_w = 120 / 12
    cell_h = 80 / 8
    for gy in range(8):
        for gx in range(12):
            n_tm = smap_full[0, gy, gx]
            n_op = smap_full[1, gy, gx]
            cx = (gx + 0.5) * cell_w
            cy = (gy + 0.5) * cell_h
            if n_tm > 0:
                ax_a.scatter([cx + (np.random.rand() - 0.5) * 4],
                             [cy + (np.random.rand() - 0.5) * 4],
                             s=110 * float(n_tm), c=COLOR_TEAMMATE,
                             marker="o", edgecolors="black", linewidths=0.6,
                             alpha=0.9, zorder=3)
            if n_op > 0:
                ax_a.scatter([cx + (np.random.rand() - 0.5) * 4],
                             [cy + (np.random.rand() - 0.5) * 4],
                             s=110 * float(n_op), c=COLOR_OPPONENT,
                             marker="o", edgecolors="black", linewidths=0.6,
                             alpha=0.9, zorder=3)
    ax_a.scatter([bx], [by], c=COLOR_BALL, marker="D", s=200,
                 edgecolors="black", linewidths=1.3, zorder=4)
    ax_a.set_title(f"(a)  Input event: Dribble by {chosen['player_name'][:24]}",
                   loc="left", fontsize=11.5, pad=8)
    # Inline legend in upper-left corner (not below plot) so caption isn't cramped
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOR_TEAMMATE,
               markeredgecolor="black", markersize=8, label="Teammate"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOR_OPPONENT,
               markeredgecolor="black", markersize=8, label="Opponent"),
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor=COLOR_BALL,
               markeredgecolor="black", markersize=8, label="Ball"),
    ]
    ax_a.legend(handles=legend_handles, loc="upper left", fontsize=9,
                frameon=True, framealpha=0.92, edgecolor="#aaa")
    ax_a.text(60, -3, "marker size ∝ players in 10×10 yd cell",
              fontsize=8.5, color="#666", ha="center", va="top", style="italic")

    # === (b) 7-channel SoccerMap montage ===
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_axis_off()
    n_ch = 7
    rows, cols = 2, 4
    for c in range(n_ch):
        r = c // cols
        col = c % cols
        # Place each channel as a small inset, leave bottom-right empty (visible padding)
        # Top row insets: y ∈ [0.50, 0.85];  bottom row: y ∈ [0.06, 0.41]
        # ~0.09 vertical gap between rows for the bottom-row inset titles.
        y0 = 0.50 if r == 0 else 0.06
        sub_ax = ax_b.inset_axes([col / cols + 0.01, y0,
                                   0.92 / cols, 0.35])
        ch = smap_full[c]
        cmap = "Blues" if c == 0 else ("Reds" if c == 1 else "viridis")
        sub_ax.imshow(ch, origin="lower", cmap=cmap, aspect="auto",
                       extent=(0, 12, 0, 8))
        sub_ax.set_xticks([]); sub_ax.set_yticks([])
        for s in sub_ax.spines.values():
            s.set_edgecolor("#888"); s.set_linewidth(0.7)
        sub_ax.set_title(CHANNEL_LABELS[c], fontsize=9.5, pad=2.5,
                          fontweight="bold", linespacing=1.0)
    ax_b.set_title("(b)  SoccerMap 7-channel encoding (8×12 grid)",
                   loc="left", fontsize=11.5, pad=8)

    # === (c) Architecture schematic ===
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_xlim(0, 10); ax_c.set_ylim(0, 10)
    ax_c.set_axis_off()

    def block(x, y, w, h, txt, fc="#ecf0f1"):
        ax_c.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                       facecolor=fc, edgecolor="#333",
                                       linewidth=0.9))
        ax_c.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                   fontsize=8.5)

    def arrow(x1, y1, x2, y2):
        ax_c.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                         arrowstyle="-|>", mutation_scale=12,
                                         color="#333", linewidth=0.9))

    block(0.4, 7.4, 4.0, 1.3, "Event features\nh_event", fc="#d6eaf8")
    block(0.4, 5.2, 4.0, 1.3, "SoccerMap CNN\nh_spatial", fc="#fadbd8")
    block(5.4, 6.2, 4.2, 1.8, "Gate  g = sigmoid(W [h_e; h_s])\n"
                                "h_fused = g h_s + (1-g) h_e",
          fc="#fef9e7")
    block(5.4, 3.4, 4.2, 1.5, "Predictor MLP\nP(success)", fc="#d5f5e3")
    block(5.4, 0.8, 4.2, 1.7, f"Output\nP = {float(chosen['p']):.2f}\n"
                                f"truth: {'success' if chosen['y_true']==1 else 'fail'}",
          fc="#fdebd0")
    arrow(4.4, 8.0, 5.4, 7.4)
    arrow(4.4, 5.8, 5.4, 6.6)
    arrow(7.5, 6.2, 7.5, 4.9)
    arrow(7.5, 3.4, 7.5, 2.5)
    ax_c.set_title("(c)  Task-Adaptive Spatial Gating",
                   loc="left", fontsize=11.5, pad=8)

    fig.suptitle("Method overview: input event  >  SoccerMap encoding  >  gated fusion  >  outcome prediction",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    png = FIG / "fig1.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig1.pdf", bbox_inches="tight")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
