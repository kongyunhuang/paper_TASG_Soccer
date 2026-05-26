#!/usr/bin/env python3
"""
compare_xpass_vs_sb: Case 3 of §5.5: our xPass vs StatsBomb pass_success_probability
========================================================================================
Ours: M4 CNN 2ch on Pass, strict holdout (24/25 test set).
StatsBomb: pass_success_probability (industrial-grade xPass 360, embedded in
L1_events_v3.parquet).

Pass is the only one of the 8 binary tasks with a built-in StatsBomb per-event
probability field, so the industrial-baseline comparison is restricted to the
Pass task.

Inputs:
  data/predictions/pass_M4_CNN_2ch_strict_with_ids.npz  (from save_pass_predictions.py)
  data/L1_events_v3.parquet (contains the StatsBomb pass_success_probability field)

Outputs:
  data/results_xpass_vs_sb.json  (ours/sb AUC/Brier/LogLoss, Pearson, bootstrap ΔAUC CI)
  figures/fig7.png       (single-panel reliability curves, ours blue vs SB red)
  figures/fig7.pdf

Figure design (redone 2026-05-18):
  The original figure had 2 panels: (a) per-event scatter + (b) reliability
  curves. Panel (a) scatter duplicated the Pearson rho=0.882 + n=98,831 already
  reported in the body, so it has been dropped. Panel (b) reliability is the
  visual-only evidence for the calibration advantage; it is kept and enlarged.
  The AUC/Brier/Log-loss numbers are still reported in the body at L473.

Usage:
  PYTHONPATH=. python scripts/plotting/compare_xpass_vs_sb.py

"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (brier_score_loss, log_loss, roc_auc_score)

# Allow direct script execution (e.g., IDE Run button) regardless of CWD.
# Adds project root to sys.path so `code.plot_utils` can be imported.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.plotting.plot_utils import set_paper_style

# Local palette: aligned with Fig 2/3 academic Paul Tol/Wong family.
COLOR_OURS = "#1F4E79"   # deep blue, matches Fig 2/3 NORMAL_COLOR
COLOR_SB   = "#CC3311"   # red, matches Fig 2/3 OUTLIER_COLOR / DUEL_COLOR
COLOR_REF  = "#888888"   # neutral grey for y=x reference line


CACHE = Path("data")
PRED = CACHE / "predictions"
FIG = Path("figures"); FIG.mkdir(exist_ok=True)


def main():
    set_paper_style()
    print("Loading our predictions ...")
    d = np.load(PRED / "pass_M4_CNN_2ch_strict_with_ids.npz", allow_pickle=True)
    df_ours = pd.DataFrame({
        "event_id": d["event_id"].astype(str),
        "y_true": d["y_true"].astype(float),
        "p_ours": d["probs"].astype(float),
    })

    print("Loading SB pass_success_probability ...")
    df_sb = pd.read_parquet(CACHE / "L1_events_v3.parquet",
                            columns=["event_id", "pass_success_probability",
                                     "pass_outcome_name", "season_dir"])
    df_sb["event_id"] = df_sb["event_id"].astype(str)

    merged = df_ours.merge(df_sb, on="event_id", how="left")
    merged = merged.dropna(subset=["pass_success_probability"])
    print(f"Merged: {len(merged):,} events with both predictions")

    y = merged["y_true"].values
    p_ours = merged["p_ours"].values
    p_sb = merged["pass_success_probability"].values

    metrics = {
        "n_events": int(len(merged)),
        "ours": {
            "auc": float(roc_auc_score(y, p_ours)),
            "brier": float(brier_score_loss(y, p_ours)),
            "log_loss": float(log_loss(y, np.clip(p_ours, 1e-6, 1 - 1e-6))),
            "mean_p": float(p_ours.mean()),
        },
        "sb": {
            "auc": float(roc_auc_score(y, p_sb)),
            "brier": float(brier_score_loss(y, p_sb)),
            "log_loss": float(log_loss(y, np.clip(p_sb, 1e-6, 1 - 1e-6))),
            "mean_p": float(p_sb.mean()),
        },
        "mean_y": float(y.mean()),
        "pearson_corr": float(np.corrcoef(p_ours, p_sb)[0, 1]),
    }
    print(f"Mean y    = {metrics['mean_y']:.4f}")
    print(f"Ours:  AUC={metrics['ours']['auc']:.4f}  Brier={metrics['ours']['brier']:.4f}  "
          f"LogLoss={metrics['ours']['log_loss']:.4f}  mean_p={metrics['ours']['mean_p']:.4f}")
    print(f"SB:    AUC={metrics['sb']['auc']:.4f}  Brier={metrics['sb']['brier']:.4f}  "
          f"LogLoss={metrics['sb']['log_loss']:.4f}  mean_p={metrics['sb']['mean_p']:.4f}")
    print(f"Corr(ours, SB) = {metrics['pearson_corr']:.4f}")

    # Paired bootstrap on AUC difference
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        a = roc_auc_score(y[idx], p_ours[idx])
        b = roc_auc_score(y[idx], p_sb[idx])
        diffs.append(a - b)
    diffs = np.array(diffs)
    metrics["bootstrap_AUC_ours_minus_sb"] = {
        "mean": float(diffs.mean()),
        "std": float(diffs.std()),
        "ci_95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_ours_better_than_sb": float((diffs > 0).mean()),
    }
    print(f"Bootstrap ΔAUC (ours - SB): {diffs.mean():+.4f} "
          f"[95% CI {np.percentile(diffs,2.5):+.4f}, {np.percentile(diffs,97.5):+.4f}]"
          f"  P(ours>SB)={(diffs>0).mean():.3f}")

    out_path = CACHE / "results_xpass_vs_sb.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    # Plot: single-panel reliability curve (scatter dropped: redundant with Pearson ρ in body)
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=500)  # matches savefig dpi=500
    n_bins = 12
    edges = np.linspace(0, 1, n_bins + 1)
    ax.plot([0, 1], [0, 1], "--", color=COLOR_REF, linewidth=0.9,
            label="Perfect calibration", zorder=1)
    for ps, color, lab in [(p_ours, COLOR_OURS, "Ours (M4 CNN 2ch)"),
                            (p_sb, COLOR_SB, "StatsBomb xPass 360")]:
        accs = []
        confs = []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i+1]
            m = (ps >= lo) & (ps < hi if i < n_bins - 1 else ps <= hi)
            if m.sum() > 30:
                accs.append(y[m].mean())
                confs.append(ps[m].mean())
        # Line + marker match Fig 2/3: white edge 0.7, lw 2.0, marker 6
        ax.plot(confs, accs, "o-", color=color, label=lab, markersize=6,
                 linewidth=2.0, markeredgecolor="white", markeredgewidth=0.7,
                 zorder=3)
    ax.set_xlabel("predicted probability", fontsize=11)
    ax.set_ylabel("empirical positive rate", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=10.5, frameon=False)
    ax.grid(False)

    # No suptitle: caption in PAPER carries description, KBS Gu 2024 convention
    fig.tight_layout()
    png = FIG / "fig7.png"
    # dpi=500 per KBS §4.1: line + halftone bitmap ≥ 500 dpi
    fig.savefig(png, dpi=500, bbox_inches="tight")
    fig.savefig(FIG / "fig7.pdf", bbox_inches="tight")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
