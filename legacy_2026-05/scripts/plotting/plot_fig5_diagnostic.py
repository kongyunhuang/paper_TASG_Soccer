#!/usr/bin/env python3
"""
plot_fig5_diagnostic: Fig 5 (6-panel 2x3) covariate-shift diagnostic main figure
======================================================================
Main figure for paper §5.4. Six sub-panels cover the full diagnostic chain:
  Row 1 (Anomaly evidence):
    (a) Per-task AUC x Brier trajectory: 8 tasks x {Majority -> B2 -> M4 -> G1};
        Dribble and Duel highlighted in red (Brier does not fall as AUC rises),
        the other 6 tasks in grey (trajectory moves down-right, Brier improves).
    (b) Channel ablation, strict 24/25: 8 tasks x 3 CNN models, ΔAUC = 7ch - 2ch heatmap.
    (c) Channel ablation, cross-league mean: 8 tasks x 2 models (M4/G1),
        same metric, heatmap with shared colour scale.
  Row 2 (Diagnostic protocol):
    (d) Reliability diagram for Dribble x M4: strict (grey, off the diagonal)
        vs relaxed (blue, hugs the diagonal).
    (e) Per-season-pair Δμ/σ for sb_num_defenders_on_goal_side: Dribble + Duel
        across 3 pairs, highlighting the 23/24 -> 24/25 jump.
    (f) Strict vs Relaxed ECE: 4 (task, model) combos x 2 splits
        (strict vs relaxed) as a grouped bar plot.

Inputs:
  data/results_calibration.json        (Panel a Brier/AUC + Panel f strict ECE)
  data/results_ablation_channels.json  (Panel b 2ch values)
  data/results_all.json                (Panel b 7ch values + Panel a baseline)
  data/results_cross_league_8tasks.json + _7ch.json (Panel c)
  data/results_season_pairs.json       (Panel e)
  data/results_relaxed_holdout.json    (Panel f relaxed ECE)
  data/predictions/dribble_M4_CNN_2ch_test_calib.npz + _relaxed.npz (Panel d)

Outputs:
  figures/fig4.{png,pdf} (14x9 inch, 300 dpi)

Usage:
  PYTHONPATH=. python scripts/plotting/plot_fig5_diagnostic.py

"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.plotting.plot_utils import set_paper_style

CACHE = _PROJECT_ROOT / "data"
PRED = CACHE / "predictions"
FIG = _PROJECT_ROOT / "figures"

# ============================================================
# Style: Tol accent palette + Arial sans-serif + 300 dpi (matches Fig 2/3)
# ============================================================
COLOR_NORMAL = "#B0B0B0"        # pale gray for 6 normal tasks
COLOR_ANOMALY = "#CC3311"       # Tol accent red for Dribble + Duel (matches Fig 3 Duel red)
COLOR_STRICT = "#888888"        # gray for strict-holdout series
COLOR_RELAXED = "#1F4E79"       # Tol blue for relaxed-holdout series (matches Fig 3 normal)
COLOR_B2 = "#5599C2"            # light blue
COLOR_M4 = "#1F4E79"            # Tol deep blue
COLOR_G1 = "#882255"            # Tol wine
COLOR_MAJ = "#A0A0A0"           # mid-gray for Majority

MODEL_COLORS = {"Majority": COLOR_MAJ, "B2": COLOR_B2, "M4": COLOR_M4, "G1": COLOR_G1}

TASKS = ["pass", "dribble", "ball_receipt", "shot", "duel",
         "interception", "ball_recovery", "pressure"]
TASK_LABELS = {
    "pass": "Pass", "dribble": "Dribble", "ball_receipt": "Ball Receipt",
    "shot": "Shot", "duel": "Duel", "interception": "Interception",
    "ball_recovery": "Ball Recovery", "pressure": "Pressure",
}
ANOMALY_TASKS = {"dribble", "duel"}

CNN_MODELS_STRICT = ["M3_CNN_Event", "M4_CNN_Full", "G1_Gating"]
CNN_MODEL_LABELS = {"M3_CNN_Event": "M3 CNN Event",
                    "M4_CNN_Full": "M4 CNN Full",
                    "G1_Gating": "G1 Gating"}
CNN_MODELS_CL = ["M4_CNN_Full", "G1_Gating"]
CNN_MODEL_LABELS_CL = {"M4_CNN_Full": "M4 CNN Full", "G1_Gating": "G1 Gating"}


def _set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333",
        "axes.unicode_minus": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# ============================================================
# Panel (a) Slope chart: Majority → M4 per task, AUC + Brier sub-panels
# ============================================================
def panel_a(ax_auc, ax_brier):
    """Slope chart per task across 4 model positions: Maj → B2 → M4 → G1.

    8 lines per sub-panel (1 line per task), connecting Majority value (left)
    to B2 (XGB+360, non-CNN baseline) to M4 (CNN concat) to G1 (CNN gating).
    Dribble + Duel highlighted red, other 6 tasks gray.

    Visual story:
      (a1) AUC: all 8 lines mostly slope UP from Maj to G1 (AUC universally
           improves with richer model). Some non-monotonicity at M4 → G1 is
           normal (different fusion architectures, not stagewise).
      (a2) Brier: 6 task lines slope DOWN (Brier improves with model richness);
           Dribble + Duel uniquely slope UP: most dramatic from M4 to G1
           (Dribble: 0.256 → 0.360 = +41% Brier worsening when CNN gating is
           applied to the same shifted distribution). This is the AUC-Brier
           disagreement that drives §5.4 diagnosis.
    """
    cal = json.loads((CACHE / "results_calibration.json").read_text())
    by_tm = {(r["task"], r["model"]): r for r in cal}

    model_ids = ["Majority", "B2_XGB_360", "M4_CNN_2ch", "G1_Gating_2ch"]
    model_xlabels = ["Maj", "B2", "M4", "G1"]
    x_positions = list(range(len(model_ids)))

    for task in TASKS:
        aucs, briers = [], []
        for m in model_ids:
            r = by_tm.get((task, m))
            if r is None:
                aucs.append(np.nan); briers.append(np.nan)
                continue
            aucs.append(r["auc"]); briers.append(r["brier"])
        if all(np.isnan(aucs)):
            continue

        is_anom = task in ANOMALY_TASKS
        color = COLOR_ANOMALY if is_anom else COLOR_NORMAL
        lw = 2.2 if is_anom else 1.2
        alpha = 1.0 if is_anom else 0.55
        zorder = 5 if is_anom else 3
        ms = 5.5 if is_anom else 4.0

        ax_auc.plot(x_positions, aucs, "-o", color=color, lw=lw, ms=ms,
                    alpha=alpha, zorder=zorder, markeredgecolor="white",
                    markeredgewidth=0.6, solid_capstyle="round")
        ax_brier.plot(x_positions, briers, "-o", color=color, lw=lw, ms=ms,
                      alpha=alpha, zorder=zorder, markeredgecolor="white",
                      markeredgewidth=0.6, solid_capstyle="round")

        # Annotate anomaly tasks at G1 (rightmost) endpoint
        if is_anom:
            g1_x = x_positions[-1]
            ax_auc.annotate(TASK_LABELS[task], xy=(g1_x, aucs[-1]),
                            xytext=(5, 0), textcoords="offset points",
                            fontsize=9.5, fontweight="bold",
                            color=COLOR_ANOMALY, va="center", zorder=6)
            ax_brier.annotate(TASK_LABELS[task], xy=(g1_x, briers[-1]),
                              xytext=(5, 0), textcoords="offset points",
                              fontsize=9.5, fontweight="bold",
                              color=COLOR_ANOMALY, va="center", zorder=6)

    # Format both sub-panels (no sub-titles; letter label "(a)" added at fig level)
    for ax, ylabel, ylim in [
        (ax_auc, "AUC", (0.45, 1.02)),
        (ax_brier, "Brier", (-0.01, 0.42)),
    ]:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(model_xlabels, fontsize=9.5)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.3, x_positions[-1] + 0.6)
        ax.set_ylim(*ylim)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", which="both", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend is added at fig level (after main() positions both sub-axes) so it
    # can sit centered below the combined (a1) + (a2) cell, horizontal layout.
    # No in-axes legend here.


# ============================================================
# Panels (b) + (c) Channel ablation heatmaps (shared color scale)
# ============================================================
def _load_strict_ablation():
    main = json.loads((CACHE / "results_all.json").read_text())
    abl = json.loads((CACHE / "results_ablation_channels.json").read_text())
    full = {(r["task"], r["model"]): r["test_auc"]
            for r in main if r.get("model") in CNN_MODELS_STRICT and "test_auc" in r}
    two = {(r["task"], r["model"].replace("_2ch", "")): r["test_auc"]
           for r in abl}
    delta = np.full((len(TASKS), len(CNN_MODELS_STRICT)), np.nan)
    for i, t in enumerate(TASKS):
        for j, m in enumerate(CNN_MODELS_STRICT):
            if (t, m) in full and (t, m) in two:
                delta[i, j] = full[(t, m)] - two[(t, m)]
    return delta


def _load_cross_league_ablation():
    rows_7 = json.loads((CACHE / "results_cross_league_8tasks_7ch.json").read_text())
    rows_2 = json.loads((CACHE / "results_cross_league_8tasks.json").read_text())

    def _short(m):
        if "M4" in m: return "M4_CNN_Full"
        if "G1" in m: return "G1_Gating"
        return None

    a7 = defaultdict(list)
    for r in rows_7:
        m = _short(r.get("model", ""))
        if m is None or r.get("task") not in TASKS: continue
        a7[(r["task"], m)].append(r["test_auc"])
    a2 = defaultdict(list)
    for r in rows_2:
        m = _short(r.get("model", ""))
        if m is None or r.get("task") not in TASKS: continue
        a2[(r["task"], m)].append(r["test_auc"])

    delta = np.full((len(TASKS), len(CNN_MODELS_CL)), np.nan)
    for i, t in enumerate(TASKS):
        for j, m in enumerate(CNN_MODELS_CL):
            if a7.get((t, m)) and a2.get((t, m)):
                delta[i, j] = float(np.mean(a7[(t, m)])) - float(np.mean(a2[(t, m)]))
    return delta


def _draw_heatmap(ax, delta, col_models, col_labels, vmax, show_y=True):
    n_rows, n_cols = delta.shape
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "tol_diverging", ["#CC3311", "#F7F7F7", "#1F4E79"], N=256)
    im = ax.imshow(delta, cmap=cmap, norm=norm, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            d = delta[i, j]
            if np.isnan(d):
                ax.text(j, i, "-", ha="center", va="center", fontsize=9, color="#777")
                continue
            # Raw ΔAUC decimal (matches sports analytics convention used by
            # Decroos VAEP, un-xPass Robberechts 2023 KDD, Gu 2024 KBS Vol.283,
            # Davis 2024: no ×100 / pp transformation).
            txt = f"{d:+.3f}"
            tc = "black" if abs(d) < vmax * 0.55 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, color=tc, fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([col_labels[m] for m in col_models], fontsize=9.5,
                       rotation=18, ha="right", rotation_mode="anchor")
    if show_y:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([TASK_LABELS[t] for t in TASKS], fontsize=10)
    else:
        ax.set_yticks(range(n_rows)); ax.set_yticklabels([])
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.6)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.6)
    # No in-panel title; letter labels added at fig level via _add_panel_label
    return im


def panel_b(ax, vmax):
    delta = _load_strict_ablation()
    _draw_heatmap(ax, delta, CNN_MODELS_STRICT, CNN_MODEL_LABELS, vmax,
                  show_y=True)


def panel_c(ax, vmax):
    delta = _load_cross_league_ablation()
    im = _draw_heatmap(ax, delta, CNN_MODELS_CL, CNN_MODEL_LABELS_CL, vmax,
                       show_y=False)
    return im


# ============================================================
# Panel (d) Reliability diagram (Dribble × M4 strict vs relaxed)
# ============================================================
def _reliability_curve(y_true, p_pred, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    pred_in_bin = np.digitize(p_pred, bin_edges[1:-1])
    bin_p, bin_y, bin_n = [], [], []
    for b in range(n_bins):
        mask = pred_in_bin == b
        if mask.sum() < 5:
            bin_p.append(np.nan); bin_y.append(np.nan); bin_n.append(0)
        else:
            bin_p.append(float(p_pred[mask].mean()))
            bin_y.append(float(y_true[mask].mean()))
            bin_n.append(int(mask.sum()))
    return bin_centers, np.asarray(bin_p), np.asarray(bin_y), np.asarray(bin_n)


def panel_d(ax):
    """Reliability diagram for Dribble × M4: strict vs relaxed holdout."""
    strict = np.load(PRED / "dribble_M4_CNN_2ch_test_calib.npz", allow_pickle=True)
    relaxed = np.load(PRED / "dribble_M4_CNN_2ch_relaxed.npz", allow_pickle=True)

    sb_c, sb_p, sb_y, sb_n = _reliability_curve(
        strict["y_true"].astype(float), strict["probs_before"].astype(float))
    rb_c, rb_p, rb_y, rb_n = _reliability_curve(
        relaxed["y_test"].astype(float), relaxed["probs"].astype(float))

    # Diagonal reference (perfect calibration)
    ax.plot([0, 1], [0, 1], "--", color="#666666", linewidth=1.0, label="Perfect calibration", zorder=2)

    # Strict curve (gray, away from diagonal)
    mask_s = ~np.isnan(sb_p)
    ax.plot(sb_p[mask_s], sb_y[mask_s], "-o", color=COLOR_STRICT,
            linewidth=1.6, markersize=4.5, alpha=0.95,
            label="Strict 22/23+23/24 → 24/25", zorder=4)

    # Relaxed curve (blue, near diagonal)
    mask_r = ~np.isnan(rb_p)
    ax.plot(rb_p[mask_r], rb_y[mask_r], "-o", color=COLOR_RELAXED,
            linewidth=1.6, markersize=4.5, alpha=0.95,
            label="Relaxed +80% of 24/25", zorder=5)

    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted probability  $\\hat{p}$")
    ax.set_ylabel("Observed positive rate  $\\bar{y}$")
    ax.set_axisbelow(True)
    # 2-item legend at upper-left (dashed diagonal needs no legend entry: it
    # is the standard "perfect calibration" reference in reliability diagrams).
    # Upper-left is empty: strict curve goes from (0, 0) climbing, relaxed
    # curve hugs the diagonal upward.
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="-", color=COLOR_STRICT,
                   lw=1.6, ms=4.5, alpha=0.95),
        plt.Line2D([0], [0], marker="o", linestyle="-", color=COLOR_RELAXED,
                   lw=1.6, ms=4.5, alpha=0.95),
    ]
    ax.legend(legend_handles,
              ["Strict 22/23+23/24 → 24/25", "Relaxed +80% of 24/25"],
              loc="upper left", framealpha=0.92, edgecolor="#bbb", fontsize=9)


# ============================================================
# Panel (e) Per-season-pair Δμ/σ for sb_num_defenders_on_goal_side
# ============================================================
def panel_e(ax):
    d = json.loads((CACHE / "results_season_pairs.json").read_text())
    feat = "sb_num_defenders_on_goal_side"
    pair_labels = ["22/23→23/24", "23/24→24/25", "22/23→24/25"]
    pair_keys = ["22/23 -> 23/24", "23/24 -> 24/25", "22/23 -> 24/25"]

    dribble = [d["tasks"]["dribble"]["per_feature"][feat][k]["delta_over_std"]
               for k in pair_keys]
    duel = [d["tasks"]["duel"]["per_feature"][feat][k]["delta_over_std"]
            for k in pair_keys]

    x = np.arange(len(pair_labels))
    bar_w = 0.36
    bars_d = ax.bar(x - bar_w/2, dribble, bar_w, color=COLOR_RELAXED, alpha=0.85,
                    edgecolor="#333", linewidth=0.6, label="Dribble", zorder=3)
    bars_u = ax.bar(x + bar_w/2, duel, bar_w, color=COLOR_G1, alpha=0.85,
                    edgecolor="#333", linewidth=0.6, label="Duel", zorder=3)

    # Stagger labels per task to avoid horizontal overlap when bars are very
    # close in height (pair 1: +0.011 / +0.020 small positive bars). Dribble
    # labels tighter to bar, Duel labels further away.
    for bars, vals, pad_pos, pad_neg in [
        (bars_d, dribble, 0.005, -0.025),
        (bars_u, duel,    0.024, -0.044),
    ]:
        for bar, v in zip(bars, vals):
            y_text = v + pad_pos if v >= 0 else v + pad_neg
            ax.text(bar.get_x() + bar.get_width()/2, y_text, f"{v:+.3f}",
                    ha="center", fontsize=9.5, fontweight="bold", color="#222")

    ax.axhline(0, color="#333", linewidth=0.7, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=9.5,
                       rotation=18, ha="right", rotation_mode="anchor")
    ax.set_ylabel("$\\Delta\\mu / \\sigma_{a}$  (b vs a)")
    ax.set_ylim(-0.50, 0.10)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", framealpha=0.92, edgecolor="#bbb")


# ============================================================
# Panel (f) Strict vs Relaxed ECE for 4 anomaly combos
# ============================================================
def panel_f(ax):
    cal = json.loads((CACHE / "results_calibration.json").read_text())
    relaxed = json.loads((CACHE / "results_relaxed_holdout.json").read_text())

    combos = [("dribble", "M4_CNN_2ch", "Dribble × M4"),
              ("dribble", "G1_Gating_2ch", "Dribble × G1"),
              ("duel", "M4_CNN_2ch", "Duel × M4"),
              ("duel", "G1_Gating_2ch", "Duel × G1")]

    strict_ece = []
    relaxed_ece = []
    for task, model, _ in combos:
        s = next(r for r in cal if r["task"] == task and r["model"] == model)
        r = next(rr for rr in relaxed if rr["task"] == task and rr["model"] == model)
        strict_ece.append(s["ece"])
        relaxed_ece.append(r["test"]["ece"])

    x = np.arange(len(combos))
    bar_w = 0.36
    bars_s = ax.bar(x - bar_w/2, strict_ece, bar_w, color=COLOR_STRICT, alpha=0.95,
                    edgecolor="#333", linewidth=0.6, label="Strict", zorder=3)
    bars_r = ax.bar(x + bar_w/2, relaxed_ece, bar_w, color=COLOR_RELAXED, alpha=0.95,
                    edgecolor="#333", linewidth=0.6, label="Relaxed", zorder=3)

    for bars, vals in [(bars_s, strict_ece), (bars_r, relaxed_ece)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.008, f"{v:.3f}",
                    ha="center", fontsize=9, fontweight="bold", color="#222")

    # ECE=0.10 reference line: practical "well-calibrated" threshold from
    # Guo 2017 etc. Annotate to the RIGHT of the last bar group in the
    # margin where the dashed line is visible (not behind any bar).
    ax.axhline(0.10, color="#999", linestyle="--", linewidth=0.9, zorder=2)
    ax.text(len(combos) - 0.5, 0.108, "ECE = 0.10",
            ha="left", va="bottom", fontsize=9.5, color="#666", style="italic",
            clip_on=False)
    ax.set_xticks(x)
    # 1-line full labels with mild rotation to fit narrow panel width and stay
    # vertically aligned with other panels' x-tick labels (1 baseline).
    ax.set_xticklabels([c[2] for c in combos], fontsize=9.5,
                       rotation=18, ha="right", rotation_mode="anchor")
    ax.set_ylabel("ECE")
    ax.set_ylim(0, 0.45)
    ax.set_axisbelow(True)
    # 2-item legend (Strict / Relaxed); upper-right is clear since rightmost
    # bars (Duel × M4 strict 0.198 / G1 0.217) top out well below the legend.
    ax.legend(loc="upper right", framealpha=0.92, edgecolor="#bbb", fontsize=9)


# ============================================================
# Main: assemble 2 × 3 grid
# ============================================================
def main():
    set_paper_style()
    _set_style()

    # Pre-compute shared vmax for (b) + (c)
    delta_s = _load_strict_ablation()
    delta_c = _load_cross_league_ablation()
    shared_vmax = float(max(np.nanmax(np.abs(delta_s)),
                            np.nanmax(np.abs(delta_c)), 0.02))

    fig = plt.figure(figsize=(10, 7), dpi=300)
    # width_ratios=[1.35, 1, 1] widens (a) cell so its two sub-panels (a1/a2)
    # each get ~1.6 inch of plot area instead of feeling squished.
    # hspace=0.32 brings row 2 closer to row 1 for more compact figure;
    # leaves just enough room for (a) shared legend below row 1 and
    # (d/e/f) letter labels above row 2 without collision.
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.32,
                          width_ratios=[1.35, 1.0, 1.0],
                          left=0.06, right=0.96, top=0.93, bottom=0.07)

    # Panel (a) cell is sub-divided into 2 slope sub-panels (a1) AUC + (a2) Brier
    # wspace=0.7 leaves clear room for (a2)'s y-axis label "Brier" so it does
    # NOT visually cross into (a1)'s plot area on the right.
    gs_a = gs[0, 0].subgridspec(1, 2, wspace=0.85)
    ax_a1 = fig.add_subplot(gs_a[0])
    ax_a2 = fig.add_subplot(gs_a[1])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    panel_a(ax_a1, ax_a2)
    panel_b(ax_b, shared_vmax)
    im_c = panel_c(ax_c, shared_vmax)
    # Shared colorbar for (b) + (c), placed on right of (c)
    cbar = fig.colorbar(im_c, ax=ax_c, fraction=0.045, pad=0.02)
    cbar.set_label("ΔAUC (7ch − 2ch)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    # Raw ΔAUC ticks (no ×100 transformation): matches heatmap cell text and
    # sports analytics convention (Decroos VAEP, un-xPass, Gu 2024 KBS, etc).

    panel_d(ax_d)
    panel_e(ax_e)
    panel_f(ax_f)

    # Shift panel (b) right and panel (c) left to tighten the (b)↔(c) gap
    # since they share the colorbar / task labels on the y-axis side.
    bbox_b = ax_b.get_position()
    ax_b.set_position([bbox_b.x0 + 0.03, bbox_b.y0,
                       bbox_b.width, bbox_b.height])
    bbox_c = ax_c.get_position()
    ax_c.set_position([bbox_c.x0 - 0.02, bbox_c.y0,
                       bbox_c.width, bbox_c.height])
    # Shift (e) left to tighten the d-e gap.
    bbox_e = ax_e.get_position()
    ax_e.set_position([bbox_e.x0 + 0.02, bbox_e.y0,
                    bbox_e.width, bbox_e.height])
    bbox_d = ax_d.get_position()
    ax_d.set_position([bbox_d.x0 + 0.05, bbox_d.y0,
                    bbox_d.width, bbox_d.height])

    # Shift row 2 (d, e, f) left by 0.02 figure-coords as a block, preserving the
    # already tuned d-e / e-f relative spacing, so that the row aligns horizontally
    # with row 1.
    for ax in [ax_d, ax_e, ax_f]:
        bbox = ax.get_position()
        ax.set_position([bbox.x0 - 0.02, bbox.y0,
                         bbox.width, bbox.height])

    # Shift row 1 panels (b) and (c) right by 0.01 figure-coords as a block,
    # preserving the already tuned b/c relative spacing.
    for ax in [ax_b, ax_c]:
        bbox = ax.get_position()
        ax.set_position([bbox.x0 + 0.01, bbox.y0,
                         bbox.width, bbox.height])

    # Minimal letter labels (a)-(f) at top-left of each panel cell, fig-level
    # for consistent positioning. Panel (a) label sits above ax_a1 (leftmost
    # sub-axis) to span the whole (a) cell visually.
    for ax, letter in [
        (ax_a1, "(a)"),
        (ax_b,  "(b)"),
        (ax_c,  "(c)"),
        (ax_d,  "(d)"),
        (ax_e,  "(e)"),
        (ax_f,  "(f)"),
    ]:
        bbox = ax.get_position()
        fig.text(bbox.x0 - 0.005, bbox.y1 + 0.015, letter,
                 fontsize=13, fontweight="bold", ha="left", va="bottom")

    # Panel (a) shared legend: horizontal layout (ncol=2) below the combined
    # (a1) + (a2) cell, centered on the cell's x-extent. Sits clear of x-axis
    # labels (Maj/B2/M4/G1) of both sub-axes.
    bbox_a1 = ax_a1.get_position()
    bbox_a2 = ax_a2.get_position()
    legend_x_center = (bbox_a1.x0 + bbox_a2.x1) / 2.0
    legend_y = bbox_a1.y0 - 0.025  # ~0.055 below x-axis labels (figure coords)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="-", color=COLOR_NORMAL,
                   lw=1.2, ms=4, alpha=0.55,
                   markeredgecolor="white", markeredgewidth=0.6),
        plt.Line2D([0], [0], marker="o", linestyle="-", color=COLOR_ANOMALY,
                   lw=2.2, ms=5.5,
                   markeredgecolor="white", markeredgewidth=0.6),
    ]
    fig.legend(legend_handles,
               ["Normal (6 tasks)", "Anomaly (Dribble, Duel)"],
               loc="upper center",
               bbox_to_anchor=(legend_x_center, legend_y),
               bbox_transform=fig.transFigure,
               ncol=2, frameon=False, fontsize=9.5,
               handletextpad=0.5, columnspacing=1.5)


    png = FIG / "fig4.png"
    pdf = FIG / "fig4.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"  Strict ablation grid: {delta_s.shape}, NaN count: {int(np.isnan(delta_s).sum())}")
    print(f"  Cross-league grid:    {delta_c.shape}, NaN count: {int(np.isnan(delta_c).sum())}")
    print(f"  Shared vmax: {shared_vmax:.4f}")


if __name__ == "__main__":
    main()
