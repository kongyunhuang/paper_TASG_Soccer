#!/usr/bin/env python3
"""
plot_main_heatmap: Fig 2 (1×2: main heatmap + family-level summary)
====================================================================
Fig 2 (1×2 multi-panel):
  Panel (a): 9 task x 10 model main heatmap (AUC for binary, R^2 for xG)
             colour scale = within-row (per task) normalisation; values annotated
             in each cell; the best value per row is framed in black.
  Panel (b): family-level summary: 6 model families (event-only / +360 /
             +CNN concat / TASG / Unified / Single-task oracle) showing the
             per-task best mean AUC across the 8 binary tasks as a horizontal bar.

Note: Fig 5 (covariate-shift diagnostic 6-panel) is produced by
   scripts/plotting/plot_fig5_diagnostic.py; the original single-panel Fig 5 output was
   removed from this script on 2026-05-16.

Inputs:
  data/results_all.json                       90 entries (9 task x 10 model)
  data/results_ablation_channels.json         24 entries (8 task x 3 model, 2ch)

Outputs:
  figures/fig2.png / .pdf       (Fig 2, 1x2 multi-panel)

Usage:
  PYTHONPATH=. python scripts/plotting/plot_main_heatmap.py

"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Rectangle

# Allow direct script execution (e.g., IDE Run button) regardless of CWD.
# Adds project root to sys.path so `code.plot_utils` can be imported.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.plotting.plot_utils import set_paper_style

CACHE = _PROJECT_ROOT / "data"
FIG = _PROJECT_ROOT / "figures"; FIG.mkdir(exist_ok=True)

TASKS = ["pass", "dribble", "ball_receipt", "shot", "duel",
         "interception", "ball_recovery", "pressure"]
TASK_LABELS = {
    "pass": "Pass",
    "dribble": "Dribble",
    "ball_receipt": "Ball Receipt",
    "shot": "Shot",
    "duel": "Duel",
    "interception": "Interception",
    "ball_recovery": "Ball Recovery",
    "pressure": "Pressure",
}
# Models ordered by family for visual progression in Fig 2
# (Table 5 keeps the original L1, L2, B1, B2... order; this reorder is figure-only)
MODELS = ["L1_LR_Event", "B1_XGB_Event", "M1_MLP_Event",      # family: event-only
          "L2_LR_360", "B2_XGB_360", "M2_MLP_360",            # family: +360 scalar
          "M3_CNN_Event", "M4_CNN_Full",                      # family: +CNN concat
          "G1_Gating",                                        # family: TASG
          "U1_Unified"]                                       # family: Unified
MODEL_LABELS = {
    "L1_LR_Event": "L1\nLR\nEvent",
    "B1_XGB_Event": "B1\nXGB\nEvent",
    "M1_MLP_Event": "M1\nMLP\nEvent",
    "L2_LR_360": "L2\nLR\n360",
    "B2_XGB_360": "B2\nXGB\n360",
    "M2_MLP_360": "M2\nMLP\n360",
    "M3_CNN_Event": "M3\nCNN\nEvent",
    "M4_CNN_Full": "M4\nCNN\nFull",
    "G1_Gating": "G1\nTASG\nGating",
    "U1_Unified": "U1\nUnified\nMulti-task",
}

# Model family color: cohesive palette (Paul Tol Muted accent for "ours")
# 3 grayscale baselines + paired wine/rose for our methods + black for oracle
# All colors CVD-safe (Wong/Tol palette family)
FAMILY_DEF = [
    ("Event only",   ["L1_LR_Event", "B1_XGB_Event", "M1_MLP_Event"], "#DCDCDC"),
    ("+ 360 scalar", ["L2_LR_360", "B2_XGB_360", "M2_MLP_360"],       "#A0A0A0"),
    ("+ CNN concat", ["M3_CNN_Event", "M4_CNN_Full"],                 "#606060"),
    ("TASG (G1)",    ["G1_Gating"],                                   "#882255"),
    ("Unified (U1)", ["U1_Unified"],                                  "#CC6677"),
]
ORACLE_COLOR = "#1A1A1A"
BEST_BORDER_COLOR = "#000000"  # neutral black, not aligned with any family

# MLP-architecture track: one specific model per family stage.
# Each x-point in panel (a) = one real model (not a family aggregate).
# LR/XGB baselines (L1, L2, B1, B2) remain in Table 5 for completeness.
# Tuple: (panel_a_short_tick, panel_b_bar_label, model_key)
MLP_TRACK = [
    ("M1", "Event\n(M1)",    "M1_MLP_Event"),
    ("M2", "+360\n(M2)",     "M2_MLP_360"),
    ("M4", "+CNN\n(M4)",     "M4_CNN_Full"),
    ("G1", "TASG\n(G1)",     "G1_Gating"),
    ("U1", "Unified\n(U1)",  "U1_Unified"),
]


def _model_family_color(model_key):
    """Look up the family color of a given model."""
    for _label, model_list, color in FAMILY_DEF:
        if model_key in model_list:
            return color
    return "#888888"

ABL_TASKS = ["pass", "dribble", "ball_receipt", "shot", "duel",
             "interception", "ball_recovery", "pressure"]
ABL_MODELS = ["M3_CNN_Event", "M4_CNN_Full", "G1_Gating"]
ABL_MODEL_LABELS = {"M3_CNN_Event": "M3 CNN Event",
                    "M4_CNN_Full": "M4 CNN Full",
                    "G1_Gating":    "G1 Gating"}

# Cross-league ablation: only M4 and G1 were run under 7ch
ABL_CL_MODELS = ["M4_CNN_Full", "G1_Gating"]
ABL_CL_MODEL_LABELS = {"M4_CNN_Full": "M4 CNN Full",
                       "G1_Gating":    "G1 Gating"}


def load_main_grid():
    """Load 8 binary tasks × 10 models AUC grid. xG row excluded (uses R²/MAE,
    not directly comparable to AUC; reported in Table 5 + prose instead)."""
    rows = json.loads((CACHE / "results_all.json").read_text())
    grid = np.full((len(TASKS), len(MODELS)), np.nan)
    raw = np.full_like(grid, np.nan)
    for r in rows:
        if r["task"] not in TASKS or r["model"] not in MODELS:
            continue
        i = TASKS.index(r["task"])
        j = MODELS.index(r["model"])
        v = r.get("test_auc", np.nan)
        grid[i, j] = v
        raw[i, j] = v
    return grid, raw


def _set_fig2_style():
    """Local rcParams for Fig 2: sans-serif (Arial/Helvetica),
    consistent with Gu 2024 KBS Vol.283 convention.
    Font sizes tuned for figsize=(11, 4.6) → typeset double-column (~6.85\")."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "mathtext.fontset": "dejavusans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#333",
        "axes.unicode_minus": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.10,
    })


def load_ablation_grid():
    """Return ΔAUC = full(7ch) − ablation(2ch) for 8 tasks × 3 models."""
    main_rows = json.loads((CACHE / "results_all.json").read_text())
    abl_rows = json.loads((CACHE / "results_ablation_channels.json").read_text())

    full = {}
    for r in main_rows:
        if r["model"] in ABL_MODELS and r["task"] in ABL_TASKS:
            full[(r["task"], r["model"])] = r["test_auc"]

    abl = {}
    for r in abl_rows:
        m = r["model"].replace("_2ch", "")
        if m in ABL_MODELS:
            abl[(r["task"], m)] = r["test_auc"]

    delta = np.full((len(ABL_TASKS), len(ABL_MODELS)), np.nan)
    raw7 = np.full_like(delta, np.nan)
    raw2 = np.full_like(delta, np.nan)
    for i, t in enumerate(ABL_TASKS):
        for j, m in enumerate(ABL_MODELS):
            if (t, m) in full and (t, m) in abl:
                delta[i, j] = full[(t, m)] - abl[(t, m)]
                raw7[i, j] = full[(t, m)]
                raw2[i, j] = abl[(t, m)]
    return delta, raw7, raw2


def load_cross_league_ablation():
    """Return ΔAUC = 7ch − 2ch averaged over EPL→LaLiga and LaLiga→EPL
    directions, for 8 tasks × {M4, G1}.

    M3 was not included in the 7ch cross-league run, so the cross-league panel
    only shows M4 and G1 columns.
    """
    rows_7 = json.loads((CACHE / "results_cross_league_8tasks_7ch.json").read_text())
    rows_2 = json.loads((CACHE / "results_cross_league_8tasks.json").read_text())

    def _short(m):
        if "M4" in m:
            return "M4_CNN_Full"
        if "G1" in m:
            return "G1_Gating"
        return None

    auc7 = {}
    for r in rows_7:
        m = _short(r.get("model", ""))
        if m is None or r.get("task") not in ABL_TASKS:
            continue
        auc7.setdefault((r["task"], m), []).append(r["test_auc"])

    auc2 = {}
    for r in rows_2:
        m = _short(r.get("model", ""))
        if m is None or r.get("task") not in ABL_TASKS:
            continue
        auc2.setdefault((r["task"], m), []).append(r["test_auc"])

    delta = np.full((len(ABL_TASKS), len(ABL_CL_MODELS)), np.nan)
    for i, t in enumerate(ABL_TASKS):
        for j, m in enumerate(ABL_CL_MODELS):
            v7 = auc7.get((t, m), [])
            v2 = auc2.get((t, m), [])
            if v7 and v2:
                delta[i, j] = float(np.mean(v7)) - float(np.mean(v2))
    return delta


def compute_slope_data(grid):
    """Per-task AUC across the 5-stage MLP architecture track.

    Returns array shape (n_tasks, 5): each column is ONE specific model:
    M1 (Event) / M2 (+360) / M4 (+CNN) / G1 (TASG) / U1 (Unified).

    LR/XGB baselines (L1, L2, B1, B2) are NOT shown in Fig 2; they remain
    in Table 5 as the precision reference.
    """
    track_models = [m for _, _, m in MLP_TRACK]
    col_idx = [MODELS.index(m) for m in track_models]
    arr = grid[:, col_idx]  # shape (n_tasks, 5)
    labels = [short for short, _, _ in MLP_TRACK]
    return arr, labels


def draw_small_multiples(axes, slope_arr, task_labels,
                         is_bottom_row, is_left_col):
    """2×4 small multiples: one panel per task, shared y-axis [0.50, 1.00].

    All 7 normal tasks use the same deep-blue line; Duel uses red dashed
    (the only inverse-direction task in §5.1 / §5.4 covariate-shift story).
    No endpoint labels (each panel has its title), no arrows, no callouts.
    """
    n_tasks, n_x = slope_arr.shape  # 8, 5
    x = np.arange(n_x)

    NORMAL_COLOR = "#1F4E79"   # deep blue (Wong-safe, distinct from cividis)
    OUTLIER_COLOR = "#CC3311"  # red
    OUTLIER_TASK = "Duel"

    # Short single-line ticks (M1, M2, M4, G1, U1); panel (b) bars show the
    # full stage name. Reader cross-references the model letter between (a)
    # and (b). Full mapping in caption.
    xtick_labels = [short for short, _, _ in MLP_TRACK]

    for idx, (ax, task_label, y_values) in enumerate(
            zip(axes, task_labels, slope_arr)):
        is_outlier = (task_label == OUTLIER_TASK)
        color = OUTLIER_COLOR if is_outlier else NORMAL_COLOR
        ls = "--" if is_outlier else "-"
        lw = 2.2 if is_outlier else 1.8
        ms = 6.0 if is_outlier else 5.5

        ax.plot(x, y_values, marker="o", markersize=ms,
                color=color, linewidth=lw, linestyle=ls,
                markeredgecolor="white", markeredgewidth=0.7,
                solid_capstyle="round", zorder=3)

        # Bold AUC value at endpoint (Unified): single number, reviewer-friendly
        ax.annotate(f"{y_values[-1]:.3f}",
                    xy=(x[-1], y_values[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=11, color=color, fontweight="bold",
                    va="center", ha="left", zorder=5)

        # Task title (top-left of each panel)
        ax.set_title(task_label, fontsize=11, fontweight="bold",
                     loc="left", pad=4,
                     color=OUTLIER_COLOR if is_outlier else "#1A1A1A")

        ax.set_ylim(0.50, 1.00)
        # Wider x-margin so tick labels (Event / +360 / +CNN / TASG / U1) breathe
        ax.set_xlim(-0.55, n_x - 0.15)
        ax.set_xticks(x)
        # Add x-axis margin between labels by widening spacing if needed
        ax.margins(x=0.08)

        # X-tick labels on every panel (top + bottom row) for self-readability
        ax.set_xticklabels(xtick_labels, fontsize=10, rotation=0)
        ax.tick_params(axis="x", which="both", length=0)

        # Y-tick labels only on leftmost column
        if is_left_col[idx]:
            ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax.set_yticklabels(["0.50", "0.60", "0.70", "0.80", "0.90", "1.00"],
                                fontsize=10)
            ax.set_ylabel("Test AUC", fontsize=10)
        else:
            ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax.set_yticklabels([])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)


def draw_main_heatmap(ax, grid, raw):
    n_rows, n_cols = grid.shape
    # Per-row normalization for color (each task scaled into [0,1] within its row)
    norm_grid = np.full_like(grid, np.nan)
    for i in range(n_rows):
        row = grid[i]
        valid = ~np.isnan(row)
        if valid.sum() == 0:
            continue
        lo, hi = np.nanmin(row), np.nanmax(row)
        if hi > lo:
            norm_grid[i] = (row - lo) / (hi - lo)
        else:
            norm_grid[i] = 0.5

    cmap = plt.get_cmap("cividis")
    im = ax.imshow(norm_grid, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Pattern figure: color encoding only for non-best cells; best per task
    # gets black border + bold value annotation. Detailed numbers in Table 5.
    for i in range(n_rows):
        row = raw[i]
        if np.all(np.isnan(row)):
            continue
        best_j = int(np.nanargmax(row))
        for j in range(n_cols):
            v = row[j]
            if np.isnan(v):
                ax.text(j, i, "-", ha="center", va="center", fontsize=9,
                        color="#777")
                continue
            if j == best_j:
                # Bold value annotation on the best cell only
                cell_norm = norm_grid[i, j]
                tc = "white" if cell_norm < 0.45 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=10, color=tc, fontweight="bold")
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor=BEST_BORDER_COLOR,
                                       linewidth=2.2, zorder=4))

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=8.5,
                       linespacing=1.1)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([TASK_LABELS[t] for t in TASKS], fontsize=10.5)
    ax.set_title("(a)  Per-task AUC pattern (8 binary tasks × 10 models; "
                 "best per task framed and labeled; full numbers in Table 5)",
                 loc="left", fontsize=11.5, pad=10)

    # Hide tick marks; keep labels
    ax.tick_params(axis="both", which="both", length=0)

    # Light grid lines between rows
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.7)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.7)

    # ── Family color band: thin horizontal bar below the heatmap ──
    # Use ax's data coordinates; bar sits just under row n_rows-0.5
    # We draw via twiny invisible axes? Simpler: use figure-level rectangle.
    # Cleaner approach: add a secondary axis under the x-tick labels with colors.
    # Instead, draw via Rectangle patches in data coords, below the bottom row.
    band_y = n_rows - 0.4   # immediately under bottom row
    band_h = 0.30
    # First, draw a thin white separator
    for label, model_list, color in FAMILY_DEF:
        col_idx = sorted(MODELS.index(m) for m in model_list)
        if not col_idx:
            continue
        x0 = col_idx[0] - 0.5
        x1 = col_idx[-1] + 0.5
        ax.add_patch(Rectangle((x0, band_y), x1 - x0, band_h,
                               facecolor=color, edgecolor="white",
                               linewidth=1.2, zorder=5, clip_on=False))
        # Family label centered on the band
        xc = (x0 + x1) / 2
        # Use white text on dark backgrounds, dark text on light
        white_text_colors = {"#606060", "#882255"}
        text_color = "white" if color in white_text_colors else "#1A1A1A"
        ax.text(xc, band_y + band_h / 2, label,
                ha="center", va="center", fontsize=9,
                color=text_color,
                fontweight="bold", zorder=6, clip_on=False)
    # Extend ax bottom so band is visible
    ax.set_ylim(band_y + band_h + 0.2, -0.5)

    # Per-row normalized colorbar (just shows the scale)
    cbar = plt.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cbar.set_label("within-task normalized AUC", fontsize=9.5)
    cbar.ax.tick_params(labelsize=9)


def compute_family_summary(grid):
    """MLP-track ladder: 5 specific models + Oracle reference.

    Each bar = mean AUC over 8 binary tasks of ONE specific model
    (M1, M2, M4, G1, U1) plus the Oracle (per-task max across all 10 models).
    Bar colors match the model's family band color in panel (a).
    """
    out = []
    for _short, bar_label, model_key in MLP_TRACK:
        col_data = grid[:, MODELS.index(model_key)]
        mean_auc = float(np.nanmean(col_data))
        color = _model_family_color(model_key)
        out.append((bar_label, mean_auc, color))
    oracle = float(np.nanmean(np.nanmax(grid, axis=1)))
    out.append(("Best per task\n(oracle)", oracle, ORACLE_COLOR))
    return out


def draw_family_summary(ax, summary):
    """Horizontal bar of family-level mean AUC over 8 binary tasks.

    Labels are short (no model-list reminder), colors match family band in (a).
    """
    labels = [s[0] for s in summary]
    values = [s[1] for s in summary]
    colors = [s[2] for s in summary]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors,
                   edgecolor="white", linewidth=0.8, height=0.72)

    # Value annotation right of each bar
    for bar, val in zip(bars, values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                fontsize=10, fontweight="bold", color="#222")

    # Reference dashed line at oracle level so reviewers see "gap to oracle"
    oracle_val = values[-1]
    ax.axvline(oracle_val, color="#222", linestyle="--", linewidth=0.8,
               alpha=0.4, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Mean AUC over 8 binary tasks", fontsize=11)
    ax.invert_yaxis()

    vmin = min(values) - 0.025
    vmax = max(values) + 0.030
    ax.set_xlim(vmin, vmax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["left"].set_color("#333")
    ax.tick_params(axis="x", labelsize=9.5)
    ax.tick_params(axis="y", which="both", length=0)
    ax.grid(axis="x", color="#EEE", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def draw_ablation_heatmap(ax, delta, col_models, col_label_map,
                          vmax, title, show_yticks=True,
                          show_ylabel=True, show_cbar=True):
    """Render one ΔAUC = 7ch − 2ch heatmap panel.

    Shared `vmax` across panels enables direct color comparison between strict
    and cross-league. Cells annotate ΔAUC in percentage points.

    Color palette aligned with Fig 2 family band and Fig 3 accent:
      negative (2ch > 7ch) → Tol accent red  #CC3311 (matches Fig 3 Duel red)
      zero                 → near-white     #F7F7F7
      positive (7ch > 2ch) → Tol accent blue #1F4E79 (matches Fig 3 normal blue)
    """
    n_rows, n_cols = delta.shape
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "tol_accent_diverging", ["#CC3311", "#F7F7F7", "#1F4E79"], N=256)
    im = ax.imshow(delta, cmap=cmap, norm=norm, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            d = delta[i, j]
            if np.isnan(d):
                ax.text(j, i, "-", ha="center", va="center", fontsize=9,
                        color="#777")
                continue
            sign = "+" if d > 0 else ""
            txt = f"{sign}{d * 100:.1f}pp"
            tc = "black" if abs(d) < vmax * 0.55 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=tc,
                    fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([col_label_map[m] for m in col_models], fontsize=9)
    if show_yticks:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([TASK_LABELS[t] for t in ABL_TASKS], fontsize=10)
    else:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([])
    if show_ylabel:
        pass
    ax.set_xlabel("Model", fontsize=10.5)
    ax.set_title(title, loc="left", fontsize=11.5, pad=6)
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.6)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.6)
    if show_cbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
        cbar.set_label("ΔAUC (7ch − 2ch)", fontsize=11)
        cbar.ax.tick_params(labelsize=8)
    return im


def main():
    set_paper_style()
    _set_fig2_style()  # local override: serif Times, larger fonts, 300 dpi
    grid, raw = load_main_grid()

    # Fig 2: 8 small multiples (left) + ladder (right)
    # No suptitle / no long descriptive titles inside the figure: Gu 2024 KBS
    # convention places those in the caption below the figure in the paper.
    fig_a = plt.figure(figsize=(10, 5), dpi=300)
    outer = fig_a.add_gridspec(1, 2, width_ratios=[6, 2], wspace=0.3,
                           left=0.04, right=0.98, top=0.92, bottom=0.14)
    left_grid = outer[0].subgridspec(2, 4, hspace=0.65, wspace=0.25)

    axes_sm = []
    for i in range(2):
        for j in range(4):
            axes_sm.append(fig_a.add_subplot(left_grid[i, j]))
    is_bottom_row = [False] * 4 + [True] * 4
    is_left_col = [(idx % 4 == 0) for idx in range(8)]

    slope_arr, _ = compute_slope_data(grid)
    task_text_labels = [TASK_LABELS[t] for t in TASKS]
    draw_small_multiples(axes_sm, slope_arr, task_text_labels,
                         is_bottom_row, is_left_col)

    # (b) MLP-track ladder
    ax_b = fig_a.add_subplot(outer[1])
    summary = compute_family_summary(grid)
    draw_family_summary(ax_b, summary)

    # Minimal panel labels: only "(a)" and "(b)" at top-left corners
    fig_a.text(0.005, 0.965, "(a)", fontsize=12, fontweight="bold")
    fig_a.text(0.745, 0.965, "(b)", fontsize=12, fontweight="bold")
    png_a = FIG / "fig2.png"
    pdf_a = FIG / "fig2.pdf"
    fig_a.savefig(png_a, dpi=300, bbox_inches="tight")
    fig_a.savefig(pdf_a, bbox_inches="tight")
    print(f"Wrote {png_a} and {pdf_a}")
    print(f"  Main grid: {grid.shape}, NaN count: {int(np.isnan(grid).sum())}")
    print("  Family summary (8-binary-task mean AUC):")
    for label, val, _ in summary:
        clean = label.replace("\n", " ")
        print(f"    {clean:35s}  {val:.4f}")

    # Fig 5 (covariate-shift diagnostic 6-panel) is produced separately by
    # scripts/plotting/plot_fig5_diagnostic.py; the original single-panel Fig 5 output
    # was removed from this script on 2026-05-16.


if __name__ == "__main__":
    main()
