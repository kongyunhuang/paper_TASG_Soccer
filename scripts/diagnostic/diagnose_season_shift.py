#!/usr/bin/env python3
"""
diagnose_season_shift: locating input-distribution drift in test season 24/25 for Dribble/Duel
================================================================================================
Background:
  The temperature-scaling experiment found the model to be well calibrated on val but with
  mean_p - mean_y = +0.22~0.27 on test.
  This indicates covariate shift (the input distribution changed; the labels did not, nor is
  CNN confidence uniformly inflated).
  This script compares the input-feature distributions of 22/23+23/24 against 24/25 to
  identify the features with the largest drift.

What it does:
  For both the Dribble and Duel tasks:
    1. Event features (location, distance, angle, duration, etc.)
    2. 360 scalar features (number of visible teammates/opponents, nearest-defender distance, etc.)
    3. SoccerMap input statistics (average number of teammates/opponents, centre-of-mass location)
    4. 360 coverage
  Reports the mean / std / KS statistics of train vs test, sorted by KS.

Usage:
  PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/diagnostic/diagnose_season_shift.py

"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from scripts.training.train_all import (
    CACHE_DIR, FULL_SEASONS, SB_360_COMMON,
    TASK_CONFIG, TEST_SEASONS, TRAIN_SEASONS,
)


TASKS = ["dribble", "duel"]


def compare_distribution(train_vals, test_vals):
    train_vals = np.asarray(train_vals, dtype=np.float64)
    test_vals = np.asarray(test_vals, dtype=np.float64)
    train_vals = train_vals[np.isfinite(train_vals)]
    test_vals = test_vals[np.isfinite(test_vals)]
    if len(train_vals) < 10 or len(test_vals) < 10:
        return None
    mean_tr, mean_te = train_vals.mean(), test_vals.mean()
    std_tr = train_vals.std() + 1e-12
    ks, p = ks_2samp(train_vals, test_vals)
    return {
        "n_train": int(len(train_vals)),
        "n_test": int(len(test_vals)),
        "mean_train": float(mean_tr),
        "mean_test": float(mean_te),
        "std_train": float(train_vals.std()),
        "std_test": float(test_vals.std()),
        "delta_mean_over_std": float((mean_te - mean_tr) / std_tr),
        "ks": float(ks),
        "ks_p": float(p),
    }


def soccermap_stats_per_event(smaps_2ch):
    """
    Given a slice of (N, 2, 8, 12) soccermaps (Ch0 teammates, Ch1 opponents),
    return per-event scalar summaries: total teammates, total opponents,
    center-of-mass x/y for each channel.
    """
    # sum over spatial dims for counts
    n_teammates = smaps_2ch[:, 0].reshape(len(smaps_2ch), -1).sum(axis=1)
    n_opponents = smaps_2ch[:, 1].reshape(len(smaps_2ch), -1).sum(axis=1)

    H, W = smaps_2ch.shape[-2:]
    yy, xx = np.mgrid[0:H, 0:W]
    xx_norm = xx / (W - 1)  # 0..1
    yy_norm = yy / (H - 1)

    def com(ch):
        total = smaps_2ch[:, ch].reshape(len(smaps_2ch), -1).sum(axis=1)
        total_safe = np.where(total > 0, total, 1.0)
        com_x = (smaps_2ch[:, ch] * xx_norm).reshape(len(smaps_2ch), -1).sum(axis=1) / total_safe
        com_y = (smaps_2ch[:, ch] * yy_norm).reshape(len(smaps_2ch), -1).sum(axis=1) / total_safe
        com_x = np.where(total > 0, com_x, np.nan)
        com_y = np.where(total > 0, com_y, np.nan)
        return com_x, com_y

    com_tm_x, com_tm_y = com(0)
    com_op_x, com_op_y = com(1)
    return {
        "sm_n_teammates": n_teammates,
        "sm_n_opponents": n_opponents,
        "sm_com_tm_x": com_tm_x,
        "sm_com_tm_y": com_tm_y,
        "sm_com_op_x": com_op_x,
        "sm_com_op_y": com_op_y,
    }


def analyse_task(df, task_name):
    cfg = TASK_CONFIG[task_name]
    df_task = df[df["type_name"] == cfg["type_name"]].copy()
    if "filter_fn" in cfg:
        df_task = cfg["filter_fn"](df_task)

    is_train = df_task["season_dir"].isin(TRAIN_SEASONS).values
    is_test = df_task["season_dir"].isin(TEST_SEASONS).values
    df_tr = df_task[is_train]
    df_te = df_task[is_test]

    report = {"task": task_name,
              "n_train": int(len(df_tr)), "n_test": int(len(df_te)),
              "has_360_train": float(df_tr["has_360"].mean()),
              "has_360_test": float(df_te["has_360"].mean()),
              "features": []}

    feats = list(cfg["event_feats"])
    feats += [f for f in SB_360_COMMON if f in df_task.columns]

    for col in feats:
        if col not in df_task.columns:
            continue
        vals_tr = pd.to_numeric(df_tr[col], errors="coerce").values
        vals_te = pd.to_numeric(df_te[col], errors="coerce").values
        cmp = compare_distribution(vals_tr, vals_te)
        if cmp is not None:
            cmp["feature"] = col
            report["features"].append(cmp)

    # SoccerMap stats
    smap_tag = cfg.get("smap_tag", task_name)
    idx_path = CACHE_DIR / f"action_soccermaps_{smap_tag}_idx.parquet"
    npy_path = CACHE_DIR / f"action_soccermaps_{smap_tag}.npy"
    if idx_path.exists() and npy_path.exists():
        idx_df = pd.read_parquet(idx_path)
        smaps = np.load(npy_path, mmap_mode="r")
        # Only use 2ch (per A1 finding)
        # Match event_ids in train and test subsets
        id_to_row = {eid: i for i, eid in enumerate(idx_df["event_id"].values)}
        tr_ids = df_tr["event_id"].values
        te_ids = df_te["event_id"].values
        tr_rows = np.array([id_to_row[e] for e in tr_ids if e in id_to_row])
        te_rows = np.array([id_to_row[e] for e in te_ids if e in id_to_row])
        # Subsample for speed if very large
        MAX = 50000
        if len(tr_rows) > MAX:
            tr_rows = np.random.default_rng(0).choice(tr_rows, MAX, replace=False)
        if len(te_rows) > MAX:
            te_rows = np.random.default_rng(0).choice(te_rows, MAX, replace=False)
        # Load sub-slices (copy from mmap)
        sm_tr = np.asarray(smaps[tr_rows][:, :2, :, :])
        sm_te = np.asarray(smaps[te_rows][:, :2, :, :])
        stats_tr = soccermap_stats_per_event(sm_tr)
        stats_te = soccermap_stats_per_event(sm_te)
        for key in stats_tr:
            cmp = compare_distribution(stats_tr[key], stats_te[key])
            if cmp is not None:
                cmp["feature"] = key
                report["features"].append(cmp)

    # Sort by |delta_mean_over_std|
    report["features"].sort(key=lambda r: -abs(r["delta_mean_over_std"]))
    return report


def main():
    print("Loading L1_events_v3.parquet ...")
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet")
    df = df[df["season_dir"].isin(FULL_SEASONS)].reset_index(drop=True)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    df["duration"] = df["duration"].fillna(0)
    df["dribble_overrun"] = df["dribble_overrun"].fillna(False).astype(int)

    all_reports = []
    for task in TASKS:
        print(f"\n=== {task.upper()} ===")
        rep = analyse_task(df, task)
        all_reports.append(rep)

        print(f"  n_train={rep['n_train']:>7}  n_test={rep['n_test']:>7}")
        print(f"  has_360_train={rep['has_360_train']:.3f}  has_360_test={rep['has_360_test']:.3f}")
        print(f"  {'feature':<35}{'mean_tr':>10}{'mean_te':>10}{'Δ/σ':>8}{'KS':>8}")
        print("  " + "-" * 73)
        for f in rep["features"][:20]:  # top 20
            print(f"  {f['feature']:<35}"
                  f"{f['mean_train']:>10.3f}"
                  f"{f['mean_test']:>10.3f}"
                  f"{f['delta_mean_over_std']:>+8.3f}"
                  f"{f['ks']:>8.3f}")

    out = CACHE_DIR / "results_season_shift.json"
    with open(out, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
