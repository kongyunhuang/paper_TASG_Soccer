#!/usr/bin/env python3
"""
plot_saliency_grid: Case 2 of §5.5: Grad-CAM 4-panel saliency grid
======================================================================
4 representative cases (confusion-matrix style, HP/LP × success/fail):
  - HP_TruePos:  high-P success (model confident + correct)
  - HP_FalsePos: high-P fail    (model confident + wrong)
  - LP_FalseNeg: low-P success  (model pessimistic + player succeeded)
  - LP_TrueNeg:  low-P fail     (model pessimistic + correct)

Each panel shows:
  - Standard pitch (white background + green pitch lines).
  - viridis colour mapping of Grad-CAM activations (normalised 0-1).
  - Blue dot = teammate / red dot = opponent (marker size proportional to the
    number of players in that cell, read directly from the 2 SoccerMap channels).
  - Yellow diamond = ball location.

Inputs:
  data/gradcam_dribble_examples.npz (from gradcam_dribble.py)
  data/L1_events_v3.parquet  (used to look up ball location + player name)

Outputs:
  figures/fig6.png / .pdf

Figure design (2026-05-18 v3 palette):
  v1: coolwarm opponent-minus-teammate density + hot saliency -> blue/red vs
      yellow/red clashes.
  v2: viridis saliency + blue/red/yellow markers -> viridis high end clashes
      with the yellow ball diamond, mid-range green clashes with the pitch
      lines, and a 4-colour gradient plus 3 markers (7 hues) is information
      overload.
  v3: Purples single-hue (white -> light purple -> dark purple) + blue teammate
      / red opponent / yellow ball; the purple heatmap has zero hue clash with
      the 3 marker colours and alpha 0.45 keeps the pitch backdrop visible.

Usage:
  PYTHONPATH=. python scripts/plotting/plot_saliency_grid.py

"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow direct script execution (e.g., IDE Run button) regardless of CWD.
# Adds project root to sys.path so `code.plot_utils` can be imported.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.plotting.plot_utils import draw_pitch, overlay_heatmap, set_paper_style

# Local marker palette: aligned with Fig 2/3 academic Paul Tol/Wong family
# (rather than Flat UI default in plot_utils). CVD-safe, desaturated, paper-grade.
MARKER_TEAMMATE = "#1F4E79"   # deep blue, matches Fig 2/3 NORMAL_COLOR
MARKER_OPPONENT = "#CC3311"   # red-orange, matches Fig 2/3 OUTLIER_COLOR / DUEL_COLOR
MARKER_BALL     = "#DDCC77"   # Paul Tol Muted sand, harmonized & distinct from red/blue

CACHE = Path("data")
FIG = Path("figures"); FIG.mkdir(exist_ok=True)


def main():
    set_paper_style()
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    p = CACHE / "gradcam_dribble_examples.npz"
    if not p.exists():
        print(f"[skip] {p} not found yet: run gradcam_dribble.py first")
        return

    d = np.load(p, allow_pickle=True)
    labels = list(d["labels"])
    smap_inputs = d["smap_inputs"]   # (n, 2, 8, 12)
    cams = d["cams"]                 # (n, 8, 12)
    ps = d["ps"]
    ys = d["ys"]
    event_ids = list(d["event_ids"])

    df_ev = pd.read_parquet(CACHE / "L1_events_v3.parquet",
                            columns=["event_id", "location_x", "location_y",
                                     "player_name", "team_name"])
    df_ev["event_id"] = df_ev["event_id"].astype(str)
    ball_xy = {}
    name_lookup = {}
    for eid in event_ids:
        row = df_ev[df_ev["event_id"] == str(eid)]
        if len(row):
            ball_xy[eid] = (float(row["location_x"].iloc[0]),
                            float(row["location_y"].iloc[0]))
            name_lookup[eid] = (row["player_name"].iloc[0],
                                row["team_name"].iloc[0])
        else:
            ball_xy[eid] = (60.0, 40.0)
            name_lookup[eid] = ("?", "?")

    n = min(len(labels), 4)  # at most 4 cases

    # Layout: 2×2 panels + slim colorbar on the right (single cbar for saliency only)
    fig = plt.figure(figsize=(10, 7), dpi=500)  # matches savefig dpi=500
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.04],
                          wspace=0.12, hspace=0.22)
    axes = []
    for r in range(2):
        for c in range(2):
            axes.append(fig.add_subplot(gs[r, c]))
    # Single colorbar spanning both rows
    cax = fig.add_subplot(gs[:, 2])

    last_heat_im = None
    for i in range(n):
        ax = axes[i]
        draw_pitch(ax)

        # Layer 1: Grad-CAM saliency, Purples sequential cmap (single-hue, no clash with markers)
        cam = cams[i]
        last_heat_im = overlay_heatmap(ax, cam, cmap="Purples", alpha=0.75,
                                        vmin=0, vmax=1)

        # Layer 2: Player position dots from SoccerMap 2 channels
        # tm_grid[r, c] = teammate count in cell, op_grid[r, c] = opponent count
        tm_grid = smap_inputs[i][0]   # channel 0: teammate count
        op_grid = smap_inputs[i][1]   # channel 1: opponent count
        for rr in range(tm_grid.shape[0]):
            for cc in range(tm_grid.shape[1]):
                cell_x = (cc + 0.5) * (120.0 / tm_grid.shape[1])
                cell_y = (rr + 0.5) * (80.0  / tm_grid.shape[0])
                if tm_grid[rr, cc] > 0:
                    ax.scatter(cell_x, cell_y, c=MARKER_TEAMMATE,
                               s=55 + 30 * float(tm_grid[rr, cc]),
                               edgecolors="white", linewidths=0.7,
                               alpha=0.95, zorder=4)
                if op_grid[rr, cc] > 0:
                    ax.scatter(cell_x, cell_y, c=MARKER_OPPONENT,
                               s=55 + 30 * float(op_grid[rr, cc]),
                               edgecolors="white", linewidths=0.7,
                               alpha=0.95, zorder=4)

        # Ball: sand diamond on top, white edge to match Fig 2/3 marker convention
        bx, by = ball_xy[event_ids[i]]
        ax.scatter([bx], [by], c=MARKER_BALL, marker="D", s=110,
                   edgecolors="white", linewidths=1.0, zorder=6)

        # Title: compact, panel label only (player name moved to caption)
        pname, team = name_lookup[event_ids[i]]
        parts = pname.split()
        if len(parts) >= 3 and len(pname) > 22:
            cand = f"{parts[0][0]}. " + " ".join(parts[1:])
            pname_disp = cand if len(cand) <= 22 else f"{parts[0][0]}. {parts[1]}"
        else:
            pname_disp = pname
        title = (f"({chr(97+i)}) {labels[i]}  ·  {pname_disp}\n"
                 f"$P$ = {ps[i]:.2f}, outcome = "
                 f"{'success' if ys[i]==1 else 'fail'}")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=6)

    if n < 4:
        for j in range(n, 4):
            axes[j].axis("off")

    # Single colorbar for saliency only: density readable from dot sizes
    cb = fig.colorbar(last_heat_im, cax=cax)
    cb.set_label("Grad-CAM attention (normalized 0–1)",
                 fontsize=10, labelpad=10)
    cb.ax.tick_params(labelsize=8.5)

    # Compact in-figure legend for dot encoding (under figure body)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=MARKER_TEAMMATE,
               markeredgecolor="white", markeredgewidth=0.7, markersize=8,
               label="Teammate (size ∝ count)"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=MARKER_OPPONENT,
               markeredgecolor="white", markeredgewidth=0.7, markersize=8,
               label="Opponent (size ∝ count)"),
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor=MARKER_BALL,
               markeredgecolor="white", markeredgewidth=1.0, markersize=8,
               label="Ball"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, fontsize=11, frameon=False,
               bbox_to_anchor=(0.45, 0.03))

    # No suptitle: caption in PAPER carries description, KBS Gu 2024 convention

    png = FIG / "fig6.png"
    # dpi=500 per KBS §4.1: line + halftone bitmap ≥ 500 dpi
    fig.savefig(png, dpi=500, bbox_inches="tight")
    fig.savefig(FIG / "fig6.pdf", bbox_inches="tight")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
