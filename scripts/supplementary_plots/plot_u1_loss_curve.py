#!/usr/bin/env python3
"""
plot_u1_loss_curve: Stage 10: U1 Unified Training Diagnostic Plot (v3)
========================================================================
Purpose: Diagnostic visualization: used to spot data / model / code anomalies,
not paper-grade convergence aesthetics.

What this plot helps catch:
  - NaN / Inf / loss spikes / divergence
  - Inverted labels (any task max AUC < 0.5)
  - Task underfitting (AUC stuck at 0.5)
  - Multi-task imbalance (one task dominates)
  - Overfitting (sustained peak-then-drop)
  - Noise floor per task (epoch-to-epoch variance, last 10 epochs)
  - Plateau epoch per task (when training stops paying off)

What this plot CANNOT catch:
  - Feature leakage / data preprocessing bugs
  - Bad train/val/test split (player leakage)
  - Label calculation errors
  - Covariate shift across seasons
  → use independent data-audit scripts for these

Layout (13.5 × 11):
  Row 0 (full width): train loss + val AUC combined (raw, no smoothing)
                      + plateau marker + best epoch star + sanity readout
  Row 1 left:  xG regression val MAE
  Row 1 right: per-task val AUC noise floor (bar chart, σ of last 10 epochs)
  Row 2-3 (2×4): 8 binary tasks val AUC, ordered easy -> hard
                 each: raw + best epoch + (max-0.5) skill gap + noise σ

Input:  data/u1_epoch_history.json
Output: figures/figS6.png + .pdf

Usage:
  PYTHONPATH=. python scripts/supplementary_plots/plot_u1_loss_curve.py

"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from scripts.plotting.plot_utils import set_paper_style

CACHE = Path("data")
FIG = Path("figures"); FIG.mkdir(exist_ok=True)

TASK_DISPLAY = {
    "pass": "Pass",
    "dribble": "Dribble",
    "ball_receipt": "Ball Receipt",
    "shot": "Shot",
    "duel": "Duel",
    "interception": "Interception",
    "ball_recovery": "Ball Recovery",
    "pressure": "Pressure",
}

TASK_COLOR = {
    "pass":          "#1f77b4",
    "ball_receipt":  "#2ca02c",
    "dribble":       "#9467bd",
    "duel":          "#8c564b",
    "shot":          "#d62728",
    "ball_recovery": "#ff7f0e",
    "interception":  "#e377c2",
    "pressure":      "#bcbd22",
}


def detect_plateau_epoch(vals: np.ndarray, threshold: float = 0.99) -> int | None:
    """First epoch where val AUC reaches threshold * max(vals)."""
    arr = np.asarray(vals, dtype=float)
    valid_mask = ~np.isnan(arr)
    if valid_mask.sum() < 3:
        return None
    target = threshold * np.nanmax(arr)
    for i, v in enumerate(arr):
        if not np.isnan(v) and v >= target:
            return i + 1
    return None


def last_n_std(vals: np.ndarray, n: int = 10) -> float:
    """Std of last n valid epochs: proxy for noise floor."""
    arr = np.asarray(vals, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) < 2:
        return 0.0
    tail = valid[-min(n, len(valid)):]
    return float(np.std(tail))


def main():
    set_paper_style()

    history_path = CACHE / "u1_epoch_history.json"
    if not history_path.exists():
        raise FileNotFoundError(
            f"{history_path} missing: run scripts/training/train_unified.py first"
        )

    with open(history_path) as f:
        h = json.load(f)

    epochs = np.arange(1, len(h["train_loss_per_epoch"]) + 1)
    train_loss = np.array(h["train_loss_per_epoch"], dtype=float)
    val_auc_combined = np.array(h["val_auc_combined_per_epoch"], dtype=float)
    val_mae_xg = np.array(h.get("val_mae_xg_per_epoch", []), dtype=float)
    best_epoch = h.get("best_epoch")
    best_auc = h.get("best_val_auc_combined")
    n_train = h.get("n_train_samples")
    n_val = h.get("n_val_samples")
    n_train_str = f"{n_train:,}" if isinstance(n_train, int) else "?"
    n_val_str = f"{n_val:,}" if isinstance(n_val, int) else "?"

    per_task_aucs = {tn: np.array(h["val_auc_per_task_per_epoch"][tn], dtype=float)
                     for tn in TASK_DISPLAY}
    task_best = {tn: float(np.nanmax(a)) for tn, a in per_task_aucs.items()}
    task_final = {tn: float(a[~np.isnan(a)][-1]) for tn, a in per_task_aucs.items()}
    task_noise = {tn: last_n_std(per_task_aucs[tn], n=10) for tn in TASK_DISPLAY}
    task_plateau = {tn: detect_plateau_epoch(per_task_aucs[tn], threshold=0.99)
                    for tn in TASK_DISPLAY}
    task_order_easy_to_hard = sorted(TASK_DISPLAY, key=lambda t: -task_best[t])

    # ── sanity checks ──
    sanity = []
    if np.any(np.isnan(train_loss)) or np.any(np.isinf(train_loss)):
        sanity.append("[FAIL] NaN/Inf in train loss")
    if np.any(np.diff(train_loss) > 0.10):
        sanity.append("[FAIL] train loss spike > 0.1 between epochs")
    inverted = [TASK_DISPLAY[tn] for tn, b in task_best.items() if b < 0.5]
    if inverted:
        sanity.append(f"[FAIL] labels possibly inverted: {','.join(inverted)}")
    if not sanity:
        sanity.append("[OK] no NaN, no spike, all tasks max AUC > 0.5")
    sanity_line = "  |  ".join(sanity)

    print(f"Loaded U1 epoch history: {len(epochs)} epochs")
    print(f"  best_epoch={best_epoch}, best_val_auc_combined={best_auc:.4f}")
    print(f"  n_train={n_train_str}, n_val={n_val_str}")
    print(f"  sanity: {sanity_line}\n")
    print(f"  {'Task':14s} {'best':>7s}  {'final':>7s}  {'σ_last10':>9s}  plateau@")
    for tn in task_order_easy_to_hard:
        plateau = task_plateau[tn]
        print(f"  {TASK_DISPLAY[tn]:14s} {task_best[tn]:>7.4f}  "
              f"{task_final[tn]:>7.4f}  {task_noise[tn]:>9.4f}  "
              f"{plateau if plateau else '-'}")

    # ── Figure ──
    fig = plt.figure(figsize=(14.0, 12.0))
    gs = gridspec.GridSpec(
        4, 4,
        height_ratios=[1.35, 1.0, 0.95, 0.95],
        hspace=0.85, wspace=0.42,
        left=0.07, right=0.96, top=0.91, bottom=0.06,
    )

    # ─── (a) Top: train loss + val AUC combined (RAW only) ───
    axT = fig.add_subplot(gs[0, :])
    c_loss = "#444444"
    c_auc = "#1f77b4"

    axT.plot(epochs, train_loss, color=c_loss, marker="o",
             markersize=4.0, linewidth=1.6, zorder=3, label="Train loss")
    axT.set_xlabel("Epoch")
    axT.set_ylabel("Train loss (avg / batch)", color=c_loss)
    axT.tick_params(axis="y", labelcolor=c_loss)
    axT.set_xlim(0.5, max(epochs) + 0.5)
    axT.grid(axis="x", alpha=0.22, linestyle=":")
    axT.spines["top"].set_visible(False)

    axT2 = axT.twinx()
    axT2.plot(epochs, val_auc_combined, color=c_auc, marker="s",
              markersize=4.5, linewidth=1.6, zorder=3, label="Val AUC combined")
    axT2.set_ylabel("Val AUC (combined 8 binary)", color=c_auc)
    axT2.tick_params(axis="y", labelcolor=c_auc)
    axT2.spines["top"].set_visible(False)

    auc_lo = float(np.nanmin(val_auc_combined))
    auc_hi = float(np.nanmax(val_auc_combined))
    auc_pad_lo = (auc_hi - auc_lo) * 0.18 + 1e-4
    auc_pad_hi = (auc_hi - auc_lo) * 0.55 + 1e-4
    axT2.set_ylim(auc_lo - auc_pad_lo, auc_hi + auc_pad_hi)

    plateau_combined = detect_plateau_epoch(val_auc_combined, threshold=0.99)
    if plateau_combined:
        axT.axvline(plateau_combined, color="#555555", linewidth=1.2,
                    linestyle=":", alpha=0.85, zorder=1)
        axT.text(plateau_combined + 0.4, axT.get_ylim()[0] +
                 (axT.get_ylim()[1] - axT.get_ylim()[0]) * 0.04,
                 f"plateau @ ep {plateau_combined}\n(99% of peak val AUC)",
                 fontsize=9.5, color="#444444", fontweight="semibold",
                 va="bottom", ha="left", alpha=0.95)

    if best_epoch:
        axT.axvline(best_epoch, color="#d62728", linewidth=1.2,
                    linestyle="--", alpha=0.85, zorder=1)
        axT2.scatter([best_epoch], [val_auc_combined[best_epoch - 1]],
                     s=120, marker="*", color="#d62728", zorder=6,
                     edgecolor="white", linewidth=0.9)
        post_best = val_auc_combined[best_epoch - 1:]
        post_spread = float(np.nanmax(post_best) - np.nanmin(post_best))
        axT2.annotate(
            f"best @ ep {best_epoch}, val AUC = {best_auc:.4f}\n"
            f"(eps {best_epoch}-30 spread = {post_spread:.4f}; "
            f"mostly noise after plateau)",
            xy=(best_epoch, val_auc_combined[best_epoch - 1]),
            xytext=(0.30, 0.93),
            textcoords="axes fraction",
            fontsize=9, color="#d62728",
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#d62728", linewidth=0.8, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="#d62728",
                            lw=0.9, alpha=0.75,
                            connectionstyle="arc3,rad=-0.20"),
        )

    axT.set_title(
        "(a) Train loss & combined val AUC vs epoch (raw, no smoothing)",
        fontsize=11, pad=8,
    )
    h1, l1 = axT.get_legend_handles_labels()
    h2, l2 = axT2.get_legend_handles_labels()
    axT.legend(h1 + h2, l1 + l2, loc="lower left",
               fontsize=8.5, framealpha=0.92, ncol=2)

    # ─── (b) Row 1 left: xG MAE ───
    axXG = fig.add_subplot(gs[1, :2])
    if len(val_mae_xg) == len(epochs):
        axXG.plot(epochs, val_mae_xg, color="#9467bd", marker="D",
                  markersize=4.0, linewidth=1.4)
        axXG.set_xlabel("Epoch")
        axXG.set_ylabel("Val MAE  (xG regression)")
        axXG.set_xlim(0.5, max(epochs) + 0.5)
        axXG.grid(axis="both", alpha=0.22, linestyle=":")
        axXG.spines["top"].set_visible(False)
        axXG.spines["right"].set_visible(False)

        if best_epoch:
            axXG.axvline(best_epoch, color="#d62728", linewidth=0.9,
                         linestyle="--", alpha=0.55)
        xg_best_ep = int(np.argmin(val_mae_xg) + 1)
        xg_best_mae = float(np.min(val_mae_xg))
        axXG.scatter([xg_best_ep], [xg_best_mae], s=85, marker="*",
                     color="#7f3fb2", zorder=5, edgecolor="white", linewidth=0.7)
        axXG.set_title(
            f"(b) xG regression val MAE  "
            f"(best={xg_best_mae:.4f} @ ep {xg_best_ep}, "
            f"final={val_mae_xg[-1]:.4f})",
            fontsize=10.5, pad=5,
        )
    else:
        axXG.text(0.5, 0.5, "no xG MAE in JSON", ha="center", va="center",
                  transform=axXG.transAxes)

    # ─── (c) Row 1 right: per-task noise bar chart ───
    axNoise = fig.add_subplot(gs[1, 2:])
    sorted_tasks_noise = sorted(TASK_DISPLAY, key=lambda t: -task_noise[t])
    noise_vals = [task_noise[t] for t in sorted_tasks_noise]
    bar_colors = [TASK_COLOR[t] for t in sorted_tasks_noise]
    bars = axNoise.barh(range(len(sorted_tasks_noise)), noise_vals,
                        color=bar_colors, alpha=0.85, edgecolor="white",
                        linewidth=0.7)
    axNoise.set_yticks(range(len(sorted_tasks_noise)))
    axNoise.set_yticklabels([TASK_DISPLAY[t] for t in sorted_tasks_noise],
                            fontsize=9)
    axNoise.invert_yaxis()
    axNoise.set_xlabel("σ of val AUC, last 10 epochs (noise floor)")
    axNoise.grid(axis="x", alpha=0.22, linestyle=":")
    axNoise.spines["top"].set_visible(False)
    axNoise.spines["right"].set_visible(False)
    for i, v in enumerate(noise_vals):
        axNoise.text(v + max(noise_vals) * 0.02, i,
                     f"{v:.4f}", va="center", fontsize=8)
    axNoise.set_xlim(0, max(noise_vals) * 1.22)
    axNoise.set_title(
        "(c) Per-task val-AUC noise (epoch-to-epoch σ, last 10 epochs)",
        fontsize=10.5, pad=5,
    )

    # ─── (d) Bottom 2×4 grid: per-task val AUC, easy -> hard ───
    for idx, tn in enumerate(task_order_easy_to_hard):
        row = 2 + idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])

        aucs = per_task_aucs[tn]
        valid = ~np.isnan(aucs)
        color = TASK_COLOR[tn]

        ax.plot(epochs[valid], aucs[valid], color=color, marker="o",
                markersize=3.5, linewidth=1.4, zorder=3)

        lo = float(np.nanmin(aucs))
        hi = float(np.nanmax(aucs))
        pad = max((hi - lo) * 0.20, 0.005)
        ax.set_ylim(lo - pad, hi + pad)

        if best_epoch and valid[best_epoch - 1]:
            ax.axvline(best_epoch, color="#d62728", linewidth=0.9,
                       linestyle="--", alpha=0.55, zorder=1)
            ax.scatter([best_epoch], [aucs[best_epoch - 1]], s=70,
                       marker="*", color="#d62728", zorder=5,
                       edgecolor="white", linewidth=0.7)

        max_auc = task_best[tn]
        gap = max_auc - 0.5
        ax.text(
            0.97, 0.05,
            f"max - .5 = {gap:.3f}\nσ$_{{10}}$ = {task_noise[tn]:.4f}",
            transform=ax.transAxes,
            fontsize=7.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="#aaaaaa", linewidth=0.5, alpha=0.88),
        )

        plateau = task_plateau[tn]
        plateau_str = f"@ep{plateau}" if plateau else "-"
        ax.set_title(
            f"{TASK_DISPLAY[tn]}  "
            f"(best={max_auc:.3f}, final={task_final[tn]:.3f}, "
            f"plateau {plateau_str})",
            fontsize=9.0, pad=4, color=color,
        )

        ax.set_xlim(0.5, max(epochs) + 0.5)
        ax.grid(axis="both", alpha=0.20, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col == 0:
            ax.set_ylabel("Val AUC")
        if row == 3:
            ax.set_xlabel("Epoch")
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(axis="both", labelsize=8.0)

    fig.text(
        0.5, 0.46,
        "(d) Per-task val AUC vs epoch  "
        "(8 binary tasks, easy -> hard by max AUC; tight y-range exposes per-task "
        "noise; star = best combined epoch; box reports max-skill gap and σ)",
        ha="center", va="center", fontsize=10.5,
    )

    fig.suptitle(
        f"U1 Unified Multi-Task: Training Diagnostic  "
        f"(30 epochs · n_train={n_train_str} · n_val={n_val_str} · "
        f"best_state @ ep {best_epoch})\n"
        f"Sanity: {sanity_line}",
        fontsize=10.5, y=0.975,
    )

    out_png = FIG / "figS6.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "figS6.pdf", bbox_inches="tight")
    print(f"\nSaved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
