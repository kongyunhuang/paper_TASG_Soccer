#!/usr/bin/env python3
"""
player_dribble_ranking: Case 1 of §5.5: player dribble-quality ranking + M4 vs G1 agreement
===========================================================================
Method (aligned with the player-level aggregation defined at the end of §4.4
of the PAPER and with RESEARCH_PLAN §4.8 Case 1):
  Dribble Quality (per player) = actual_success_rate - predicted_success_rate
  Positive = above the spatial expectation; negative = below it.
  Threshold: n_dribble >= 20 (matching the end of §4.4 in the PAPER).

Inputs:
  data/predictions/dribble_M4_CNN_2ch_relaxed.npz
  data/predictions/dribble_G1_Gating_2ch_relaxed.npz
  data/L1_events_v3.parquet

Outputs:
  data/results_dribble_ranking.json (top20/bot20 for both M4 and G1)
  figures/fig5.png  (single-panel M4 x G1 quality scatter + diagonal)
  figures/fig5.pdf

Figure design (redone 2026-05-18):
  The previous 2-panel bar chart (top-20 per model) was redundant with Table 9
  (top 10) in the PAPER. It has been replaced by a single-panel M4 x G1 scatter
  with a y=x diagonal, visualising Spearman 0.966 directly (the primary metric
  for the ranking use case). Pearson is omitted because it is collinear with
  Spearman when the scatter is tight along the diagonal with no outlier.
  Six anchor players are labelled (shared top-5 plus the swapped 5th-place
  pair Gravenberch/Mitoma). The file name dribble_ranking_top20.* is kept for
  a stable PAPER cross-reference (content changed, file name unchanged).

Note:
  Predictions use the relaxed holdout (not the strict one); see
  covariate_shift_diagnosis for the rationale: the strict holdout has severe
  covariate shift on Dribble, so the residual is not a real quality gap.
  The relaxed holdout has ECE < 0.10, so the player-level ranking is reliable.

Usage:
  PYTHONPATH=. python scripts/plotting/player_dribble_ranking.py

"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CACHE = Path("data")
PRED = CACHE / "predictions"
FIG = Path("figures"); FIG.mkdir(exist_ok=True)


def load_with_ids(task, model):
    d = np.load(PRED / f"{task}_{model}_relaxed.npz", allow_pickle=True)
    return pd.DataFrame({
        "event_id": d["event_id"].astype(str),
        "match_id": d["match_id"],
        "y_true": d["y_test"].astype(float),
        "y_prob": d["probs"].astype(float),
    })


def short_name(full):
    """Shorten very long player names for scatter labels.
    - If full name > 22 chars and has ≥ 3 words → abbreviate first name to initial.
    - If still > 22 chars → keep only first 2 components.
    Examples (anchor players in this script):
      'Adama Traoré Diarra'             -> unchanged (19 chars)
      'Sávio Moreira de Oliveira'       -> 'S. Moreira de Oliveira'
      'Jeremy Doku' / 'Jadon Sancho'    -> unchanged (≤ 22 chars)
    """
    parts = full.split()
    if len(parts) < 3 or len(full) <= 22:
        return full
    abbr = f"{parts[0][0]}. " + " ".join(parts[1:])
    if len(abbr) <= 22:
        return abbr
    return f"{parts[0][0]}. {parts[1]}"


def main():
    print("Loading event metadata ...")
    df_events = pd.read_parquet(CACHE / "L1_events_v3.parquet",
                                columns=["event_id", "match_id", "player_id",
                                         "player_name", "team_name", "position_name",
                                         "season_dir"])
    df_events["event_id"] = df_events["event_id"].astype(str)

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    rankings_out = {}
    ranked_by_model = {}  # collect per-model ranked DataFrames for cross-model scatter

    for model_name, label in [
        ("M4_CNN_2ch", "M4 CNN"),
        ("G1_Gating_2ch", "G1 Gating"),
    ]:
        preds = load_with_ids("dribble", model_name)
        merged = preds.merge(df_events, on="event_id", how="left")
        print(f"\n{label}: merged {len(merged):,} events "
              f"({merged['player_id'].notna().sum():,} with player meta)")

        # Drop events without player_id
        merged = merged.dropna(subset=["player_id"])

        # Aggregate by player
        agg = merged.groupby("player_id").agg(
            player_name=("player_name", "first"),
            team_name=("team_name", "first"),
            n=("y_true", "size"),
            actual_rate=("y_true", "mean"),
            predicted_rate=("y_prob", "mean"),
        ).reset_index()
        agg["quality"] = agg["actual_rate"] - agg["predicted_rate"]

        # Filter to players with minimum attempts
        MIN_N = 20
        ranked = agg[agg["n"] >= MIN_N].sort_values("quality", ascending=False).reset_index(drop=True)
        print(f"  {len(ranked)} players with ≥{MIN_N} dribbles")

        top20 = ranked.head(20).copy()
        bot20 = ranked.tail(20).copy()

        rankings_out[model_name] = {
            "min_n": MIN_N,
            "total_players_ranked": int(len(ranked)),
            "top20": top20.to_dict(orient="records"),
            "bot20": bot20.to_dict(orient="records"),
        }
        ranked_by_model[model_name] = ranked.set_index("player_id")[["player_name", "team_name", "n", "quality"]]

        # Print to console
        print(f"\n  {label} TOP 10:")
        for _, r in top20.head(10).iterrows():
            print(f"    {r['player_name']:<28} {r['team_name']:<22} "
                  f"n={int(r['n']):>3}  actual={r['actual_rate']:.3f}  "
                  f"pred={r['predicted_rate']:.3f}  Δ={r['quality']:+.3f}")
        print(f"\n  {label} BOTTOM 10:")
        for _, r in bot20.tail(10).iloc[::-1].iterrows():
            print(f"    {r['player_name']:<28} {r['team_name']:<22} "
                  f"n={int(r['n']):>3}  actual={r['actual_rate']:.3f}  "
                  f"pred={r['predicted_rate']:.3f}  Δ={r['quality']:+.3f}")

    # --- Single-panel scatter: M4 vs G1 quality on shared player set ---
    # Purpose: visualize Spearman 0.966 agreement directly (ranking-use-case primary metric)
    # Replaces previous 2-panel bar chart which was redundant with Table 9 (Top 10)
    # Pearson omitted: collinear with Spearman when scatter tight near y=x diagonal (no outlier)
    from scipy.stats import spearmanr

    m4 = ranked_by_model["M4_CNN_2ch"].rename(columns={"quality": "q_M4"})
    g1 = ranked_by_model["G1_Gating_2ch"].rename(columns={"quality": "q_G1"})
    joint = m4[["player_name", "team_name", "n", "q_M4"]].join(g1[["q_G1"]], how="inner")
    print(f"\nJoint players (M4 ∩ G1): {len(joint)}")

    sp_rho, sp_p = spearmanr(joint["q_M4"], joint["q_G1"])
    rank_m4 = joint["q_M4"].rank()
    rank_g1 = joint["q_G1"].rank()
    rank_diff_abs = (rank_m4 - rank_g1).abs()
    print(f"  Spearman ρ = {sp_rho:.4f}  (p={sp_p:.2e})")
    print(f"  Rank-diff: max={int(rank_diff_abs.max())}, mean={rank_diff_abs.mean():.2f}, "
          f"# players with rank_diff=0: {int((rank_diff_abs==0).sum())}/{len(joint)}")

    # Save Spearman + rank-diff stats to JSON for traceability (reviewer / future audit)
    rankings_out["comparison_M4_vs_G1"] = {
        "n_joint_players": int(len(joint)),
        "spearman_rho": float(sp_rho),
        "spearman_p": float(sp_p),
        "rank_diff_max": int(rank_diff_abs.max()),
        "rank_diff_mean": float(rank_diff_abs.mean()),
        "rank_diff_zero_count": int((rank_diff_abs == 0).sum()),
    }

    # dpi matches savefig dpi (500) to avoid figure/savefig DPI mismatch in text bbox calc
    fig, ax = plt.subplots(figsize=(7, 6), dpi=500)
    # Diagonal y=x reference; extend right side for long labels (Adama Traoré Diarra ~19 chars)
    # +0.18 right padding ensures full label fits inside axes before colorbar (figsize=(7,6))
    lo = min(joint["q_M4"].min(), joint["q_G1"].min()) - 0.02
    hi = max(joint["q_M4"].max(), joint["q_G1"].max()) + 0.18
    ax.plot([lo, hi], [lo, hi], "--", color="#888", linewidth=0.9, zorder=1, label="y = x")
    # Scatter, color by mean quality (red = below expectation, green = above)
    # Smaller marker (s=45) + alpha=0.75: pairs within 0.005 data units (Romero/Rogers/
    # Raul Moro cluster in bot-left, Yamal/Salah in center, Gordon/Palmer, etc.) will
    # render as DARKER stacked spots, visually signaling overlap without losing point identity.
    mean_q = 0.5 * (joint["q_M4"].values + joint["q_G1"].values)
    sc = ax.scatter(joint["q_M4"], joint["q_G1"],
                    c=mean_q, cmap="RdYlGn", s=45, alpha=0.75,
                    edgecolors="#222", linewidth=0.7, zorder=3, vmin=-0.15, vmax=0.15)
    # Label notable players. Per-player explicit offset (dx, dy, ha, va): fan out
    # radially in the upper-right cluster to avoid Sancho↔Mitoma and Yamal↔Salah overlap.
    # Yamal + Salah are LABELED because their markers physically overlap (Δq < 0.005);
    # labeling both makes the underlying 2-point pair visible.
    label_offsets = {
        "Adama Traoré Diarra":         (+0.008, +0.000, "left",   "center"),
        "Sávio Moreira de Oliveira":   (+0.008, +0.000, "left",   "center"),
        "Jeremy Doku":                 (+0.008, +0.000, "left",   "center"),
        "Jadon Sancho":                (+0.008, +0.001, "left",   "bottom"),
        "Ryan Gravenberch":            (+0.008, -0.005, "left",   "top"),
        "Kaoru Mitoma":                (-0.098, +0.015, "left",   "bottom"),  # up + right (clear of cluster)
        "Mohamed Salah":               (-0.008, +0.005, "right",  "bottom"),  # up + left (separate from Yamal)
        "Lamine Yamal Nasraoui Ebana": (+0.026, -0.016, "right",  "top"),     # down + left (clear of Salah)
    }
    for _, r in joint.iterrows():
        if r["player_name"] in label_offsets:
            short = short_name(r["player_name"])
            dx, dy, ha, va = label_offsets[r["player_name"]]
            label_xy = (r["q_M4"] + dx, r["q_G1"] + dy)
            marker_xy = (r["q_M4"], r["q_G1"])
            # Leader line only if label is significantly offset from marker
            if abs(dy) > 0.012:
                ax.annotate(short, xy=marker_xy, xytext=label_xy,
                            fontsize=9, color="#222",
                            horizontalalignment=ha, verticalalignment=va,
                            arrowprops=dict(arrowstyle="-", color="#999",
                                            linewidth=0.5, shrinkA=0, shrinkB=4))
            else:
                ax.text(label_xy[0], label_xy[1], short,
                        fontsize=9, color="#222",
                        horizontalalignment=ha, verticalalignment=va)
    # Explicit cluster annotation for bot-left 3-point fuse cluster
    # (Romero / Rogers / Raul Moro at d ≈ 0.002-0.006 in lower-left, smaller than
    # any reasonable marker diameter → 3 markers visually fuse into 1 dot. Annotation
    # makes the underlying 3-point count visible to readers counting markers.)
    cluster_x = (-0.0997 + -0.0991 + -0.0944) / 3
    cluster_y = (-0.1298 + -0.1280 + -0.1316) / 3
    ax.annotate("Romero / Rogers / Raul Moro",
                xy=(cluster_x, cluster_y),
                xytext=(cluster_x - 0.02, cluster_y - 0.01),
                fontsize=8, color="#555",
                horizontalalignment="left", verticalalignment="top",
                arrowprops=dict(arrowstyle="-", color="#999",
                                linewidth=0.5, shrinkA=0, shrinkB=4))
    ax.set_xlabel("M4 CNN 2ch :  dribble quality residual", fontsize=11)
    ax.set_ylabel("G1 Spatial Gating :  dribble quality residual", fontsize=11)
    # Annotate stats in upper-left corner (Spearman only: ranking-use-case primary metric)
    # p < 0.001 follows KBS convention (Gu 2024, un-xPass): sci notation 1e-18 not used in prior
    ax.text(0.03, 0.97,
            f"n = {len(joint)} players (n_dribble ≥ {MIN_N}, 24/25 relaxed holdout)\n"
            f"Spearman ρ = {sp_rho:.3f}   (p < 0.001)",
            transform=ax.transAxes, fontsize=10, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#888", linewidth=0.6, alpha=0.9))
    ax.axhline(0, color="#aaa", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#aaa", linewidth=0.5, zorder=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.set_axisbelow(True)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.06)  # pad 0.04→0.06: move right 2pp to clear right-edge labels (Adama)
    cb.set_label("mean quality (M4, G1)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # No suptitle: caption in PAPER carries description, KBS Gu 2024 convention
    fig.tight_layout()
    png = FIG / "fig5.png"
    # dpi=500 per KBS §4.1: line + halftone bitmap ≥ 500 dpi
    fig.savefig(png, dpi=500, bbox_inches="tight")
    fig.savefig(FIG / "fig5.pdf", bbox_inches="tight")
    print(f"\nWrote {png}")

    out = CACHE / "results_dribble_ranking.json"
    with open(out, "w") as f:
        json.dump(rankings_out, f, indent=2, default=str)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
