#!/usr/bin/env python3
"""
plot_utils: Shared figure style and pitch-drawing utilities for the paper
==============================================================================
Provides unified pitch rendering, fonts and colours across all case-study
figures so the plots look as if they come from one source.

Main exports:
  draw_pitch(ax, ...)         Draw a standard football pitch (StatsBomb 120x80 yards).
  overlay_freeze_frame(...)   Draw a 360 freeze-frame (players + ball) on the pitch.
  overlay_heatmap(...)        Overlay an 8x12 grid heatmap on the pitch (alpha blend).
  set_paper_style()           Apply the unified matplotlib rcParams.

Conventions:
  - StatsBomb coordinates: pitch (120, 80); attacking direction is to the right.
  - matplotlib axes: x in [0, 120], y in [0, 80].
  - Player markers: teammate circle (blue), opponent circle (red),
    actor square (black), ball diamond (yellow).

Usage example:
  from scripts.plotting.plot_utils import draw_pitch, overlay_heatmap
  fig, ax = plt.subplots()
  draw_pitch(ax)
  overlay_heatmap(ax, cam_8x12, cmap="viridis", alpha=0.5)

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Rectangle


PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

# Unified palette (paper-friendly, viridis-aware).
COLOR_PITCH_LINE = "#2c3e50"
COLOR_PITCH_FILL = "#fafafa"
COLOR_TEAMMATE = "#2980b9"   # blue
COLOR_OPPONENT = "#c0392b"   # red
COLOR_ACTOR = "#000000"      # actor in black
COLOR_KEEPER = "#f39c12"     # orange
COLOR_BALL = "#f1c40f"       # yellow


def set_paper_style():
    """Call at top of plotting scripts to unify look-and-feel."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def draw_pitch(ax, line_color=COLOR_PITCH_LINE, fill_color=COLOR_PITCH_FILL,
               linewidth=1.0, show_axes=False):
    """
    Draw a StatsBomb-coordinate (120×80) football pitch on ax.

    Origin is bottom-left; attacking goal at x=120.
    """
    # Pitch fill
    ax.add_patch(Rectangle((0, 0), PITCH_LENGTH, PITCH_WIDTH,
                            facecolor=fill_color, edgecolor=line_color,
                            linewidth=linewidth, zorder=0))

    # Halfway line
    ax.plot([60, 60], [0, 80], color=line_color, linewidth=linewidth, zorder=1)

    # Centre circle + spot
    ax.add_patch(Circle((60, 40), 9.15, fill=False, color=line_color,
                         linewidth=linewidth, zorder=1))
    ax.add_patch(Circle((60, 40), 0.5, color=line_color, zorder=1))

    # Penalty boxes
    ax.add_patch(Rectangle((0, 18), 18, 44, fill=False, edgecolor=line_color,
                            linewidth=linewidth, zorder=1))
    ax.add_patch(Rectangle((102, 18), 18, 44, fill=False, edgecolor=line_color,
                            linewidth=linewidth, zorder=1))

    # 6-yard boxes
    ax.add_patch(Rectangle((0, 30), 6, 20, fill=False, edgecolor=line_color,
                            linewidth=linewidth, zorder=1))
    ax.add_patch(Rectangle((114, 30), 6, 20, fill=False, edgecolor=line_color,
                            linewidth=linewidth, zorder=1))

    # Penalty spots
    ax.add_patch(Circle((12, 40), 0.4, color=line_color, zorder=1))
    ax.add_patch(Circle((108, 40), 0.4, color=line_color, zorder=1))

    # Penalty arcs
    ax.add_patch(Arc((12, 40), 18.3, 18.3, angle=0, theta1=-53, theta2=53,
                      color=line_color, linewidth=linewidth, zorder=1))
    ax.add_patch(Arc((108, 40), 18.3, 18.3, angle=180, theta1=-53, theta2=53,
                      color=line_color, linewidth=linewidth, zorder=1))

    # Goals
    ax.add_patch(Rectangle((-1.5, 36), 1.5, 8, fill=False, edgecolor=line_color,
                            linewidth=linewidth, zorder=1))
    ax.add_patch(Rectangle((120, 36), 1.5, 8, fill=False, edgecolor=line_color,
                            linewidth=linewidth, zorder=1))

    ax.set_xlim(-3, 123)
    ax.set_ylim(-3, 83)
    ax.set_aspect("equal")
    if not show_axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)


def overlay_freeze_frame(ax, freeze_frame, ball_xy=None,
                         show_keeper=True, marker_size=110):
    """
    Draw 360 freeze-frame on pitch.

    freeze_frame: list of dicts [{"location": [x, y], "teammate": bool,
                                  "actor": bool, "keeper": bool}]
                  or numpy array (N, 4) [x, y, teammate, keeper] (no actor info).
    ball_xy: (x, y) ball location.
    """
    if isinstance(freeze_frame, list):
        for ff in freeze_frame:
            x, y = ff["location"]
            is_tm = ff.get("teammate", False)
            is_actor = ff.get("actor", False)
            is_keeper = ff.get("keeper", False)
            if is_actor:
                color = COLOR_ACTOR
                marker = "s"
                size = marker_size * 1.1
                edge = "white"
            elif is_keeper:
                color = COLOR_KEEPER
                marker = "o"
                size = marker_size
                edge = COLOR_PITCH_LINE
            elif is_tm:
                color = COLOR_TEAMMATE
                marker = "o"
                size = marker_size
                edge = COLOR_PITCH_LINE
            else:
                color = COLOR_OPPONENT
                marker = "o"
                size = marker_size
                edge = COLOR_PITCH_LINE
            if (not is_keeper) or show_keeper:
                ax.scatter([x], [y], c=color, s=size, marker=marker,
                           edgecolors=edge, linewidths=0.7, zorder=3)
    else:
        # numpy fallback
        for x, y, is_tm, is_keeper in freeze_frame:
            color = COLOR_TEAMMATE if is_tm else COLOR_OPPONENT
            ax.scatter([x], [y], c=color, s=marker_size, marker="o",
                       edgecolors=COLOR_PITCH_LINE, linewidths=0.7, zorder=3)

    if ball_xy is not None:
        bx, by = ball_xy
        ax.scatter([bx], [by], c=COLOR_BALL, s=marker_size * 1.0, marker="D",
                   edgecolors="black", linewidths=0.9, zorder=4)


def overlay_heatmap(ax, grid, cmap="viridis", alpha=0.55,
                    grid_h=8, grid_w=12, vmin=None, vmax=None):
    """
    Overlay an 8×12 (default) numerical grid as a heatmap on the pitch.

    grid is plotted with extent matching the pitch (120 × 80).
    """
    if grid.shape != (grid_h, grid_w):
        raise ValueError(f"grid shape {grid.shape} != ({grid_h}, {grid_w})")
    return ax.imshow(grid, extent=(0, PITCH_LENGTH, 0, PITCH_WIDTH),
                     origin="lower", cmap=cmap, alpha=alpha, aspect="auto",
                     vmin=vmin, vmax=vmax, zorder=2, interpolation="bilinear")


def overlay_arrow(ax, start_xy, end_xy, color="#16a085", linewidth=2.0,
                  alpha=0.8, **kwargs):
    """Draw a directional arrow on the pitch."""
    sx, sy = start_xy
    ex, ey = end_xy
    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                 arrowprops=dict(arrowstyle="->", color=color,
                                 linewidth=linewidth, alpha=alpha,
                                 mutation_scale=18),
                 zorder=3, **kwargs)


def add_legend(ax, loc="upper right", fontsize=8):
    """Standard legend for player markers."""
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOR_TEAMMATE,
               markeredgecolor=COLOR_PITCH_LINE, markersize=8, label="Teammate"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOR_OPPONENT,
               markeredgecolor=COLOR_PITCH_LINE, markersize=8, label="Opponent"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOR_KEEPER,
               markeredgecolor=COLOR_PITCH_LINE, markersize=8, label="Keeper"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor=COLOR_ACTOR,
               markeredgecolor="white", markersize=8, label="Actor"),
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor=COLOR_BALL,
               markeredgecolor="black", markersize=7, label="Ball"),
    ]
    ax.legend(handles=handles, loc=loc, fontsize=fontsize, framealpha=0.9)
