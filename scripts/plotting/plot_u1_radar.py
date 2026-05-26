#!/usr/bin/env python3
"""
plot_u1_radar: Case 4 of §5.5: U1 Multi-task player profile (radar)
======================================================================
Aggregates U1 per-event predictions into a per-(player, task) quality residual:
  quality = actual_rate - predicted_rate  (per task per player on the 24/25 holdout)
  z-score = (quality - league_mean) / league_std   (standardised against the
            per-task league distribution)

Plots a 9-task radar for 6 players in a 2-row x 3-column grid (top row EPL,
bottom row La Liga, role-matched):
  EPL row:     Adama Traoré (winger)  / Kevin De Bruyne (midfielder) / Virgil van Dijk (CB)
  La Liga row: Lamine Yamal (winger)  / Jude Bellingham (midfielder) / Antonio Rüdiger (CB)

Thresholds (aligned with the end of §4.4 of the PAPER):
  LEAGUE_MIN_N = 20: league mean/std uses only players with >= 20 events.
  MIN_N_PER_TASK = 5: a player-task cell with < 5 events is shown as an x
                      on the radar and is not interpreted.

Inputs:
  data/predictions/u1_per_task.npz (produced by train_unified.py)
  data/L1_events_v3.parquet (player_id, player_name, team_name)
Outputs:
  data/results_u1_player_profiles.json
  figures/fig8.png + .pdf

Figure design (redone 2026-05-18):
  - The original 1x3 grid (3 EPL players only) is now a 2x3 grid that adds 3
    role-matched La Liga players, verifying the deployment pattern across the
    two leagues.
  - Radial ticks include +-1σ, +-3σ and +-5σ (the original +-2σ visually clipped
    a z=+5.08 outlier).
  - Extreme outliers (|z| >= 2.5) were previously annotated inline as
    "z=X.XX (n=N)".

Usage:
  PYTHONPATH=. python scripts/plotting/plot_u1_radar.py

"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from scripts.plotting.plot_utils import set_paper_style

CACHE = Path("data")
PRED = CACHE / "predictions"
FIG = Path("figures"); FIG.mkdir(exist_ok=True)

# Order of the 9 tasks (matches train_unified TASK_DEFS).
TASK_NAMES = ["pass", "dribble", "ball_receipt", "shot", "duel",
              "interception", "ball_recovery", "pressure", "xg"]
TASK_LABELS = ["Pass", "Dribble", "Ball receipt", "Shot", "Duel",
               "Interception", "Ball recov.", "Pressure", "xG"]

# 6 players: top row EPL 3 (winger / midfielder / defender), bottom row La Liga 3
# (role-matched).
TARGET_PLAYERS = [
    # EPL row
    ("Adama Traoré Diarra",     "Winger"),
    ("Kevin De Bruyne",         "Midfielder"),
    ("Virgil van Dijk",         "Defender"),
    # La Liga row
    ("Lamine Yamal Nasraoui Ebana", "Winger"),
    ("Jude Bellingham",         "Midfielder"),
    ("Antonio Rüdiger",         "Defender"),
]
MIN_N_PER_TASK = 5     # A player needs >= 5 events in a task to count (otherwise NaN).
LEAGUE_MIN_N = 20      # Player threshold for computing the league average.


def main():
    set_paper_style()
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

    print("Loading U1 per-event predictions ...")
    npz_path = PRED / "u1_per_task.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} missing: run train_unified.py first")
    d = np.load(npz_path, allow_pickle=True)
    df = pd.DataFrame({
        "event_id": d["event_id"].astype(str),
        "task_id": d["task_id"].astype(int),
        "is_regression": d["is_regression"].astype(int),
        "y_true": d["y_true"].astype(np.float32),
        "y_prob": d["y_prob"].astype(np.float32),
    })
    df["task_name"] = df["task_id"].map(lambda i: TASK_NAMES[i])
    print(f"  n_events={len(df):,}  n_tasks={df['task_id'].nunique()}")
    for t in TASK_NAMES:
        n = (df["task_name"] == t).sum()
        print(f"    {t:15s}  n={n:7,}")

    print("Loading player info ...")
    df_ev = pd.read_parquet(CACHE / "L1_events_v3.parquet",
                            columns=["event_id", "player_id", "player_name",
                                     "team_name", "position_name"])
    df_ev["event_id"] = df_ev["event_id"].astype(str)
    df = df.merge(df_ev, on="event_id", how="left")

    # ── per-(player, task) quality ──
    print("Aggregating per-(player, task) quality ...")
    df["err"] = df["y_true"] - df["y_prob"]   # quality = actual - predicted
    grp = df.groupby(["player_id", "player_name", "team_name",
                      "position_name", "task_name"])
    agg = grp.agg(n=("err", "size"),
                  mean_err=("err", "mean"),
                  actual=("y_true", "mean"),
                  predicted=("y_prob", "mean")).reset_index()
    agg = agg[agg["n"] >= MIN_N_PER_TASK].reset_index(drop=True)
    print(f"  total (player, task) rows ≥ {MIN_N_PER_TASK}: {len(agg)}")

    # ── league avg per task (for shading) ──
    # For each task: take players with n >= LEAGUE_MIN_N, mean +- 1 std of mean_err.
    league = {}
    for t in TASK_NAMES:
        sub = agg[(agg["task_name"] == t) & (agg["n"] >= LEAGUE_MIN_N)]
        if len(sub):
            league[t] = (float(sub["mean_err"].mean()),
                          float(sub["mean_err"].std(ddof=1)),
                          int(len(sub)))
        else:
            league[t] = (0.0, 0.0, 0)
    print("\nLeague avg (n≥{}):".format(LEAGUE_MIN_N))
    for t in TASK_NAMES:
        m, s, n = league[t]
        print(f"  {t:15s}  mean={m:+.4f}  std={s:.4f}  n_players={n}")

    # ── Extract per-player profiles ──
    target_profiles = []
    for player_name, role in TARGET_PLAYERS:
        sub = agg[agg["player_name"] == player_name]
        if len(sub) == 0:
            print(f"  WARN: '{player_name}' not found in U1 test set")
            continue
        # take per-task quality
        per_task = {}
        for t in TASK_NAMES:
            row = sub[sub["task_name"] == t]
            if len(row):
                per_task[t] = {
                    "n": int(row.iloc[0]["n"]),
                    "quality": float(row.iloc[0]["mean_err"]),
                    "actual": float(row.iloc[0]["actual"]),
                    "predicted": float(row.iloc[0]["predicted"]),
                }
            else:
                per_task[t] = None
        team = sub.iloc[0]["team_name"]
        pos = sub.iloc[0]["position_name"]
        target_profiles.append({
            "player_name": player_name,
            "role_label": role,
            "team_name": team,
            "position_name": pos,
            "per_task": per_task,
        })
        print(f"\n{player_name} ({role}) [{team}, pos={pos}]:")
        for t in TASK_NAMES:
            d_t = per_task[t]
            if d_t:
                print(f"  {t:15s}  n={d_t['n']:4d}  quality={d_t['quality']:+.3f}  "
                      f"actual={d_t['actual']:.3f}  pred={d_t['predicted']:.3f}")
            else:
                print(f"  {t:15s}  (n < {MIN_N_PER_TASK})")

    # ── Save JSON ──
    out_json = {
        "min_n_per_task": MIN_N_PER_TASK,
        "league_min_n": LEAGUE_MIN_N,
        "league_avg": {t: {"mean": m, "std": s, "n_players": n}
                        for t, (m, s, n) in league.items()},
        "target_profiles": target_profiles,
    }
    out_path = CACHE / "results_u1_player_profiles.json"
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2, default=str)
    print(f"\nSaved {out_path}")

    # ── Draw the 3-panel radar ──
    n_p = len(target_profiles)
    if n_p == 0:
        print("No profiles to plot: abort")
        return

    # Layout: 2 rows × 3 cols when 6 players (EPL top row + La Liga bottom row)
    if n_p == 6:
        fig, axes2d = plt.subplots(2, 3, figsize=(14.0, 10.4), dpi=500,
                                   subplot_kw={"projection": "polar"},
                                   gridspec_kw={"wspace": 0.32, "hspace": 0.42})
        axes = list(axes2d.flatten())
    else:
        fig, axes = plt.subplots(1, n_p, figsize=(5.0 * n_p, 5.6), dpi=500,
                                 subplot_kw={"projection": "polar"},
                                 gridspec_kw={"wspace": 0.40})
        if n_p == 1:
            axes = [axes]

    n_tasks = len(TASK_NAMES)
    angles = np.linspace(0, 2 * np.pi, n_tasks, endpoint=False).tolist()
    angles += angles[:1]   # close the loop

    # Standardise (z-score relative to league): (quality - league_mean) / league_std.
    # This sets "0 = league average" and makes positive/negative deviations
    # equally salient with a visually symmetric axis.
    def normalize(q, t):
        m, s, _ = league[t]
        if s < 1e-6:
            return 0.0
        return (q - m) / s

    # Pick the axis range.
    all_q_n = []
    for prof in target_profiles:
        for t in TASK_NAMES:
            d_t = prof["per_task"].get(t)
            if d_t is not None:
                all_q_n.append(normalize(d_t["quality"], t))
    qmax = max(2.0, max(abs(min(all_q_n)), abs(max(all_q_n))) * 1.15)

    # ── Colour map ──
    colors = ["#c0392b", "#2980b9", "#27ae60"]    # red / blue / green: 3 players.

    for ax_idx, (ax, prof) in enumerate(zip(axes, target_profiles)):
        # League shading: after standardisation the +-1σ band is [-1, +1].
        league_lo = [-1.0] * len(TASK_NAMES)
        league_hi = [+1.0] * len(TASK_NAMES)
        league_mean_norm = [0.0] * len(TASK_NAMES)
        league_lo += league_lo[:1]
        league_hi += league_hi[:1]
        league_mean_norm += league_mean_norm[:1]

        ax.fill_between(angles, league_lo, league_hi,
                        color="#888", alpha=0.18,
                        label="League ±1 std (n≥%d)" % LEAGUE_MIN_N)
        ax.plot(angles, league_mean_norm, color="#666", linewidth=0.9,
                linestyle="--", alpha=0.65,
                label="League mean (z=0)")

        # player series (normalized z-score)
        ply_q = []
        for t in TASK_NAMES:
            d_t = prof["per_task"].get(t)
            if d_t is not None:
                ply_q.append(normalize(d_t["quality"], t))
            else:
                ply_q.append(0.0)
        ply_q += ply_q[:1]

        col = colors[ax_idx % len(colors)]
        ax.plot(angles, ply_q, color=col, linewidth=2.4,
                marker="o", markersize=7, markeredgecolor="black",
                markeredgewidth=0.8, label=prof["player_name"])
        ax.fill(angles, ply_q, color=col, alpha=0.15)

        # NaN markers (insufficient task samples): larger + Paul Tol red for visibility.
        for i, t in enumerate(TASK_NAMES):
            d_t = prof["per_task"].get(t)
            if d_t is None:
                ax.plot([angles[i]], [0], marker="x", markersize=15,
                        markeredgecolor="#CC3311", markerfacecolor="none",
                        markeredgewidth=2.4, zorder=5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(TASK_LABELS, fontsize=11)
        ax.set_ylim(-qmax, qmax)
        # Show ticks in z units (including +-3σ and +-5σ so outliers align visually).
        ticks = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]
        ticks = [t for t in ticks if abs(t) <= qmax]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:+.0f}σ" for v in ticks], fontsize=9.5)
        # Move radial tick labels off the default 0° (Pass axis) to ~95° (gap
        # between Ball receipt and Shot), which is usually clear of polygon data.
        ax.set_rlabel_position(95)
        ax.tick_params(axis="x", pad=8)
        ax.grid(True, linestyle=":", alpha=0.45)

        # Outlier z-value annotations removed: radar spike length already conveys
        # outlier magnitude visually; specific z values + n are reported in body prose.

        # title (player name + role + team)
        title = (f"{prof['player_name']}\n"
                 f"{prof['role_label']}: {prof['team_name']}")
        ax.set_title(title, fontsize=12.5, fontweight="bold", pad=22,
                     color="#1a1a1a")

    # global legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor="#888", alpha=0.25, label="League ±1 std (n≥%d players)" % LEAGUE_MIN_N),
        Line2D([0], [0], color="#666", linewidth=1, linestyle="--",
               label="League mean"),
        Line2D([0], [0], marker="x", linestyle="",
               markeredgecolor="#CC3311", markersize=12, markeredgewidth=2.4,
               label="Insufficient n (<%d events)" % MIN_N_PER_TASK),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, 0.04))

    # No suptitle: caption in PAPER carries description, KBS Gu 2024 convention

    fig.tight_layout(rect=[0, 0.06, 1, 0.99])

    png = FIG / "fig8.png"
    # dpi=500 per KBS §4.1: line + halftone bitmap ≥ 500 dpi
    fig.savefig(png, dpi=500, bbox_inches="tight")
    fig.savefig(FIG / "fig8.pdf", bbox_inches="tight")
    print(f"Saved {png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
