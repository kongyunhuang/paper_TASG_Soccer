#!/usr/bin/env python3
"""
diagnose_season_pairs: pairwise season comparison to locate the year in which drift occurs
============================================================================================
The previous script compared 22/23+23/24 combined against 24/25 and found drift.
This script splits the three complete seasons into three pairwise comparisons:
    22/23 vs 23/24   (tests whether drift accumulates each year)
    23/24 vs 24/25   (tests whether drift concentrates at this boundary -> pipeline change)
    22/23 vs 24/25   (global drift)

Computes statistics on the key drifted features for Dribble and Duel.

Output (added from 2026-05-16):
    data/results_season_pairs.json
    Stores mean_a, mean_b, delta_mu/sigma_a, KS for every task x feature x pair, along
    with SoccerMap CoM per-season summary, so that the numbers in section 5.4 Step 4
    (e.g. the 23/24->24/25 jump of -0.32) can be cited directly.

Usage:
  PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/diagnostic/diagnose_season_pairs.py

"""

import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from scripts.training.train_all import CACHE_DIR, FULL_SEASONS, TASK_CONFIG
from scripts.diagnostic.diagnose_season_shift import soccermap_stats_per_event


# Features of interest (top drifting in the previous step)
DRIBBLE_FEATS = [
    "sb_num_defenders_on_goal_side",
    "sb_visible_teammates",
    "sb_visible_opponents",
    "sb_distance_to_nearest_defender",
    "location_x",
]
DUEL_FEATS = [
    "sb_num_defenders_on_goal_side",
    "sb_visible_teammates",
    "sb_visible_opponents",
    "sb_distance_to_nearest_defender",
    "location_x",
]

SEASONS = {
    "22/23": ["2_235_2022_23", "11_235_2022_23"],
    "23/24": ["2_281_2023_24", "11_281_2023_24"],
    "24/25": ["2_317_2024_25", "11_317_2024_25"],
}


def season_means(df, seasons_dirs):
    return df[df["season_dir"].isin(seasons_dirs)]


def compare_pair(df_a, df_b, col):
    a = pd.to_numeric(df_a[col], errors="coerce").dropna().values
    b = pd.to_numeric(df_b[col], errors="coerce").dropna().values
    if len(a) < 10 or len(b) < 10:
        return None
    ks, _ = ks_2samp(a, b)
    std_a = a.std() + 1e-12
    return {
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "delta_over_std": float((b.mean() - a.mean()) / std_a),
        "ks": float(ks),
    }


def soccermap_com_x_by_season(df_task, season_dirs, idx_lookup, smaps):
    df_s = df_task[df_task["season_dir"].isin(season_dirs)]
    ids = [e for e in df_s["event_id"].values if e in idx_lookup]
    if not ids:
        return None
    rows = np.array([idx_lookup[e] for e in ids])
    if len(rows) > 50000:
        rows = np.random.default_rng(0).choice(rows, 50000, replace=False)
    sm = np.asarray(smaps[rows][:, :2, :, :])
    stats = soccermap_stats_per_event(sm)
    return {k: v for k, v in stats.items()}


def main():
    print("Loading ...")
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet")
    df = df[df["season_dir"].isin(FULL_SEASONS)].reset_index(drop=True)

    pairs = [("22/23", "23/24"), ("23/24", "24/25"), ("22/23", "24/25")]
    out = {"tasks": {}}

    for task, feats in [("dribble", DRIBBLE_FEATS), ("duel", DUEL_FEATS)]:
        cfg = TASK_CONFIG[task]
        df_t = df[df["type_name"] == cfg["type_name"]].copy()
        out["tasks"][task] = {"per_feature": {}, "soccermap_per_season": {}}

        print(f"\n{'='*72}\n{task.upper()}: pairwise season comparison\n{'='*72}")
        for feat in feats:
            if feat not in df_t.columns:
                continue
            out["tasks"][task]["per_feature"][feat] = {}
            print(f"\n  Feature: {feat}")
            print(f"  {'pair':<18}{'mean_a':>10}{'mean_b':>10}{'Δ/σa':>8}{'KS':>8}")
            for a, b in pairs:
                df_a = season_means(df_t, SEASONS[a])
                df_b = season_means(df_t, SEASONS[b])
                r = compare_pair(df_a, df_b, feat)
                if r is None:
                    continue
                pair_label = f"{a} -> {b}"
                out["tasks"][task]["per_feature"][feat][pair_label] = r
                print(f"  {a + ' → ' + b:<18}"
                      f"{r['mean_a']:>10.3f}{r['mean_b']:>10.3f}"
                      f"{r['delta_over_std']:>+8.3f}{r['ks']:>8.3f}")

        # SoccerMap COM x
        smap_tag = cfg.get("smap_tag", task)
        idx_path = CACHE_DIR / f"action_soccermaps_{smap_tag}_idx.parquet"
        npy_path = CACHE_DIR / f"action_soccermaps_{smap_tag}.npy"
        if not (idx_path.exists() and npy_path.exists()):
            continue
        idx_df = pd.read_parquet(idx_path)
        smaps = np.load(npy_path, mmap_mode="r")
        idx_lookup = {eid: i for i, eid in enumerate(idx_df["event_id"].values)}

        print(f"\n  SoccerMap-derived (sampled ≤50K per season):")
        for label, dirs in SEASONS.items():
            stats = soccermap_com_x_by_season(df_t, dirs, idx_lookup, smaps)
            if stats is None:
                continue
            n_tm = float(stats["sm_n_teammates"].mean())
            n_op = float(stats["sm_n_opponents"].mean())
            com_tm_x = float(np.nanmean(stats["sm_com_tm_x"]))
            com_op_x = float(np.nanmean(stats["sm_com_op_x"]))
            out["tasks"][task]["soccermap_per_season"][label] = {
                "n_tm": n_tm, "n_op": n_op,
                "com_tm_x": com_tm_x, "com_op_x": com_op_x,
            }
            print(f"    {label}: n_tm={n_tm:.2f}  n_op={n_op:.2f}  "
                  f"com_tm_x={com_tm_x:.3f}  com_op_x={com_op_x:.3f}")

    out_path = CACHE_DIR / "results_season_pairs.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
