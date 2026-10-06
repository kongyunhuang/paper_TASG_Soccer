#!/usr/bin/env python3
"""
audit_label_leakage: Stage 10 diagnostic
==========================================
Per-task label leakage scan for the U1 Unified Multi-Task dataset.

Motivation:
  In Stage 10 loss-curve diagnostics we observed Ball Receipt / Dribble val AUC
  saturating to 0.99 within 1-2 epochs. That is suspicious. Either the task is
  trivial, or one or more features directly encode the label.

What this script does:
  For each of the 8 binary tasks (replicating exactly the labels and features
  used in scripts/training/train_unified.py):
    1. Build the train-split subset only (no val/test).
    2. For each feature column f_i, compute:
         max-direction ROC-AUC = max( AUC(y, X[:, i]),  AUC(y, -X[:, i]) )
         |Pearson corr| with y
    3. Sort features by AUC; print top-5 most informative single features.
    4. Flag tasks where the top-1 feature has AUC > 0.90 -> suspicious leakage.
    5. Fit a 1-feature logistic regression on the train-set top feature to
       confirm AUC remains high (i.e. the signal is monotone, not noise).
    6. Report class prevalence per task (very imbalanced -> easy AUC).
    7. Cross-check NaN-rate per feature per task.

  Also produces a Markdown summary at: audit/label_leakage_report.md

Note:
  This only checks tabular event features (~22 dims). It does NOT check the
  SoccerMap 7-channel spatial input - that would need a separate per-channel
  patch-level audit. SoccerMap is freeze-frame-at-event-start, generally
  considered pre-action context, not future info.

Input:  data/L1_events_v3.parquet
        (same parquet train_unified.py loads)
Output: audit/label_leakage_report.md
        + stdout summary table

Usage:
  PYTHONPATH=. python scripts/audit/audit_label_leakage.py

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CACHE = Path("data")
AUDIT = Path("audit"); AUDIT.mkdir(exist_ok=True)

# Replicate train_unified.py exactly
FULL_SEASONS = [
    "2_235_2022_23", "2_281_2023_24", "2_317_2024_25",
    "11_235_2022_23", "11_281_2023_24", "11_317_2024_25",
]
TRAIN_SEASONS = ["2_235_2022_23", "2_281_2023_24",
                 "11_235_2022_23", "11_281_2023_24"]

UNIFIED_EVENT_FEATS = [
    "location_x", "location_y", "dist_to_goal",
    "angle_to_goal", "angle_to_goal_center",
    "under_pressure", "duration",
    "pass_length", "pass_angle",
    "pass_end_dist_to_goal", "pass_end_angle_to_goal",
    "pass_is_progressive", "pass_lateral_displacement",
    "pass_cross", "pass_switch", "pass_through_ball", "pass_cut_back",
    "dribble_overrun",
]
SB_360 = [
    "sb_distance_to_nearest_defender", "sb_num_defenders_on_goal_side",
    "sb_visible_teammates", "sb_visible_opponents",
]
ALL_FEATS = UNIFIED_EVENT_FEATS + SB_360

ON_BALL_TYPES = {
    "Pass", "Dribble", "Shot", "Ball Receipt*", "Carry",
    "Clearance", "Interception", "Ball Recovery", "Block",
    "Foul Committed", "Foul Won", "Duel", "Dribbled Past",
    "Dispossessed", "Miscontrol", "Goal Keeper", "Pressure",
}

# Tasks (replicate train_unified.py TASK_DEFS; only binary tasks audited here)
BINARY_TASKS = {
    "pass": {
        "type_name": "Pass",
        "label_fn": lambda df: df["pass_outcome_name"].isna().astype(int).values,
        "filter_fn": None,
    },
    "dribble": {
        "type_name": "Dribble",
        "label_fn": lambda df: (df["dribble_outcome_name"] == "Complete").astype(int).values,
        "filter_fn": None,
    },
    "ball_receipt": {
        "type_name": "Ball Receipt*",
        "label_fn": lambda df: df["ball_receipt_outcome_name"].isna().astype(int).values,
        "filter_fn": None,
    },
    "shot": {
        "type_name": "Shot",
        "label_fn": lambda df: (df["shot_outcome_name"] == "Goal").astype(int).values,
        "filter_fn": lambda df: df[df["shot_type_name"] != "Penalty"],
    },
    "duel": {
        "type_name": "Duel",
        "label_fn": lambda df: df["duel_outcome_name"].isin({"Won", "Success In Play"}).astype(int).values,
        "filter_fn": None,
    },
    "interception": {
        "type_name": "Interception",
        "label_fn": lambda df: df["interception_outcome_name"].isin({"Won", "Success In Play"}).astype(int).values,
        "filter_fn": None,
    },
    "ball_recovery": {
        "type_name": "Ball Recovery",
        "label_fn": lambda df: (df["ball_recovery_failure"] != True).astype(int).values,
        "filter_fn": None,
    },
    # pressure handled separately: label computed post-hoc from next 2 on-ball events
}


def compute_pressure_labels(df: pd.DataFrame) -> np.ndarray:
    """Replicates train_unified.compute_pressure_labels()."""
    df = df.reset_index(drop=True)
    pressure_indices = df.index[df["type_name"] == "Pressure"].values
    all_types = df["type_name"].values
    all_poss = df["possession_team_id"].values
    all_match = df["match_id"].values
    is_onball = np.array([t in ON_BALL_TYPES for t in all_types])
    labels = np.zeros(len(pressure_indices), dtype=int)
    for i, pidx in enumerate(pressure_indices):
        mid = all_match[pidx]
        team = all_poss[pidx]
        onball_count = 0
        for offset in range(1, 20):
            ni = pidx + offset
            if ni >= len(df):
                break
            if all_match[ni] != mid:
                break
            if not is_onball[ni]:
                continue
            onball_count += 1
            if all_poss[ni] != team:
                labels[i] = 1
                break
            if onball_count >= 2:
                break
    return labels


def safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Return max(AUC(y, s), AUC(y, -s)). Handles NaN in scores."""
    mask = ~np.isnan(scores)
    if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
        return float("nan")
    s = scores[mask]
    yy = y[mask]
    try:
        auc_pos = roc_auc_score(yy, s)
        auc_neg = roc_auc_score(yy, -s)
        return float(max(auc_pos, auc_neg))
    except Exception:
        return float("nan")


def abs_corr(y: np.ndarray, scores: np.ndarray) -> float:
    mask = ~np.isnan(scores)
    if mask.sum() < 50:
        return float("nan")
    s = scores[mask].astype(np.float64)
    yy = y[mask].astype(np.float64)
    if s.std() < 1e-10 or yy.std() < 1e-10:
        return float("nan")
    return float(abs(np.corrcoef(s, yy)[0, 1]))


def audit_task(task_name: str, cfg: dict, df_full: pd.DataFrame,
               train_mask_full: np.ndarray,
               feat_cols: list[str]) -> dict:
    print(f"\n  ── Auditing {task_name} ──")

    df_task = df_full[df_full["type_name"] == cfg["type_name"]].copy()
    if cfg["filter_fn"] is not None:
        df_task = cfg["filter_fn"](df_task)

    if task_name == "pressure":
        labels = compute_pressure_labels(df_full)
        # need to re-filter df_task to just Pressure rows
        df_task = df_full[df_full["type_name"] == "Pressure"].reset_index(drop=True)
    else:
        labels = cfg["label_fn"](df_task)

    # Apply train-season filter
    season_mask = df_task["season_dir"].isin(TRAIN_SEASONS).values
    df_task_tr = df_task[season_mask].reset_index(drop=True)
    labels_tr = labels[season_mask]

    n = len(df_task_tr)
    if n < 100:
        print(f"    SKIP: only {n} train samples")
        return {"task": task_name, "n_train": n, "skipped": True}

    pos_rate = float(labels_tr.mean())
    print(f"    n_train = {n:,}  | pos_rate = {pos_rate:.4f}")

    # Feature-by-feature AUC
    rows = []
    for f in feat_cols:
        if f not in df_task_tr.columns:
            rows.append({"feat": f, "auc": float("nan"),
                         "abs_corr": float("nan"),
                         "nan_rate": 1.0, "n_unique": 0})
            continue
        col = df_task_tr[f].values
        try:
            col_f = col.astype(np.float64)
        except (ValueError, TypeError):
            col_f = pd.to_numeric(col, errors="coerce").values.astype(np.float64)

        nan_rate = float(np.isnan(col_f).mean())
        n_unique = int(pd.Series(col).nunique(dropna=True))
        auc = safe_auc(labels_tr, col_f)
        ac = abs_corr(labels_tr, col_f)
        rows.append({
            "feat": f, "auc": auc, "abs_corr": ac,
            "nan_rate": nan_rate, "n_unique": n_unique,
        })

    res = pd.DataFrame(rows).sort_values("auc", ascending=False)
    top5 = res.head(5)
    print("    Top-5 single-feature AUC:")
    for _, r in top5.iterrows():
        auc_s = f"{r['auc']:.4f}" if not np.isnan(r['auc']) else "  nan "
        corr_s = f"{r['abs_corr']:.3f}" if not np.isnan(r['abs_corr']) else " nan "
        print(f"      {r['feat']:38s}  AUC={auc_s}  |corr|={corr_s}  "
              f"nan_rate={r['nan_rate']:.2f}")

    top_feat = top5.iloc[0]
    flagged = (not np.isnan(top_feat["auc"])) and top_feat["auc"] > 0.90

    # Train 1-feature logistic regression as sanity check
    lr_auc = float("nan")
    if not np.isnan(top_feat["auc"]):
        f_name = top_feat["feat"]
        col = pd.to_numeric(df_task_tr[f_name], errors="coerce").values.astype(np.float64)
        mask = ~np.isnan(col)
        if mask.sum() > 100 and len(np.unique(labels_tr[mask])) == 2:
            X1 = col[mask].reshape(-1, 1)
            y1 = labels_tr[mask]
            try:
                lr = LogisticRegression(max_iter=200)
                lr.fit(X1, y1)
                lr_auc = roc_auc_score(y1, lr.predict_proba(X1)[:, 1])
            except Exception as e:
                print(f"    LR failed: {e}")

    flag_str = "  *** FLAG (top AUC > 0.90) ***" if flagged else ""
    print(f"    -> top-1 feature: {top_feat['feat']}  "
          f"AUC={top_feat['auc']:.4f}  LR AUC={lr_auc:.4f}{flag_str}")

    return {
        "task": task_name,
        "n_train": int(n),
        "pos_rate": pos_rate,
        "top5": top5.to_dict(orient="records"),
        "flagged": flagged,
        "top_feat": top_feat["feat"],
        "top_auc": float(top_feat["auc"]) if not np.isnan(top_feat["auc"]) else None,
        "top_lr_auc": float(lr_auc) if not np.isnan(lr_auc) else None,
        "skipped": False,
    }


def main():
    print("=" * 80)
    print("Stage 10 audit: label leakage scan (U1 Unified)")
    print("=" * 80)

    parquet_path = CACHE / "L1_events_v3.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} missing")
    df = pd.read_parquet(parquet_path)

    # restrict to FULL_SEASONS
    df = df[df["season_dir"].isin(FULL_SEASONS)].reset_index(drop=True)
    print(f"Loaded {len(df):,} rows from {parquet_path.name}")
    print(f"  train seasons (used here): {TRAIN_SEASONS}")

    feat_cols = [f for f in ALL_FEATS if f in df.columns]
    missing_feats = [f for f in ALL_FEATS if f not in df.columns]
    if missing_feats:
        print(f"  WARNING: {len(missing_feats)} features missing from parquet: "
              f"{missing_feats}")
    print(f"  features audited: {len(feat_cols)}\n")

    train_mask_full = df["season_dir"].isin(TRAIN_SEASONS).values

    all_results = []
    for task_name, cfg in BINARY_TASKS.items():
        try:
            r = audit_task(task_name, cfg, df, train_mask_full, feat_cols)
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR auditing {task_name}: {e}")

    # pressure task (special)
    pressure_cfg = {
        "type_name": "Pressure",
        "label_fn": None,
        "filter_fn": None,
    }
    try:
        r = audit_task("pressure", pressure_cfg, df, train_mask_full, feat_cols)
        all_results.append(r)
    except Exception as e:
        print(f"  ERROR auditing pressure: {e}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Task':14s}  {'n_train':>9s}  {'pos_rate':>9s}  "
          f"{'top feat':<32s}  {'top AUC':>8s}  {'LR AUC':>7s}  Flag")
    flagged_tasks = []
    for r in all_results:
        if r.get("skipped"):
            print(f"{r['task']:14s}  (skipped)")
            continue
        top_auc = r["top_auc"]
        lr_auc = r["top_lr_auc"]
        flag = "*** FLAG ***" if r["flagged"] else ""
        if r["flagged"]:
            flagged_tasks.append(r["task"])
        top_auc_s = f"{top_auc:.4f}" if top_auc else "  nan "
        lr_auc_s = f"{lr_auc:.4f}" if lr_auc else "  nan "
        print(f"{r['task']:14s}  {r['n_train']:>9,}  {r['pos_rate']:>9.4f}  "
              f"{r['top_feat']:<32s}  {top_auc_s:>8s}  {lr_auc_s:>7s}  {flag}")

    print()
    if flagged_tasks:
        print(f"⚠ {len(flagged_tasks)} task(s) flagged: {flagged_tasks}")
        print("   → top single feature has AUC > 0.90.")
        print("   → suggests possible leakage or trivially-imbalanced label.")
    else:
        print("no single feature reaches AUC > 0.90 on any task.")

    # ── Write markdown report ──
    md_path = AUDIT / "label_leakage_report.md"
    with open(md_path, "w") as f:
        f.write("# Label Leakage Audit: U1 Unified Multi-Task\n\n")
        f.write("**Date**: 2026-05-11  \n")
        f.write("**Script**: `scripts/audit/audit_label_leakage.py`  \n")
        f.write(f"**Train seasons**: {', '.join(TRAIN_SEASONS)}  \n\n")

        f.write("## Method\n\n")
        f.write("For each binary task, compute per-feature `max-direction ROC-AUC` "
                "= max(AUC(y, x_i), AUC(y, -x_i)) on the **train split only**.\n\n"
                "A feature with single-feature AUC > 0.90 is **flagged** as a "
                "candidate leakage source (the task can be solved with one column).\n\n")

        f.write("## Summary table\n\n")
        f.write("| Task | n_train | pos_rate | top feature | top AUC | 1-feat LR AUC | Flag |\n")
        f.write("|---|---:|---:|---|---:|---:|---|\n")
        for r in all_results:
            if r.get("skipped"):
                continue
            top_auc = r["top_auc"]
            lr_auc = r["top_lr_auc"]
            top_auc_s = f"{top_auc:.4f}" if top_auc else "nan"
            lr_auc_s = f"{lr_auc:.4f}" if lr_auc else "nan"
            flag = "🚩 FLAG" if r["flagged"] else "OK"
            f.write(f"| {r['task']} | {r['n_train']:,} | {r['pos_rate']:.4f} "
                    f"| `{r['top_feat']}` | {top_auc_s} | {lr_auc_s} | {flag} |\n")

        f.write("\n## Per-task top-5 single-feature AUC\n\n")
        for r in all_results:
            if r.get("skipped"):
                continue
            f.write(f"### {r['task']}  (pos_rate = {r['pos_rate']:.4f}, "
                    f"n_train = {r['n_train']:,})\n\n")
            f.write("| Feature | AUC | \\|corr\\| | NaN rate | n_unique |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for top in r["top5"]:
                auc_s = f"{top['auc']:.4f}" if not np.isnan(top["auc"]) else "nan"
                corr_s = f"{top['abs_corr']:.3f}" if not np.isnan(top["abs_corr"]) else "nan"
                f.write(f"| `{top['feat']}` | {auc_s} | {corr_s} "
                        f"| {top['nan_rate']:.2f} | {top['n_unique']} |\n")
            f.write("\n")

        f.write("## Interpretation cheat-sheet\n\n")
        f.write("- **top AUC > 0.95** → near-certain leakage; one feature solves the task.\n")
        f.write("- **0.90 ≤ top AUC ≤ 0.95** → likely leakage OR strong single signal "
                "(e.g. `pass_end_dist_to_goal` may legitimately predict pass success "
                "if it records intended target, illegitimately if it records actual end).\n")
        f.write("- **top AUC ≤ 0.90** → no single-column leakage; AUC comes from "
                "feature combinations and spatial map.\n")
        f.write("- **pos_rate > 0.95** → highly imbalanced; even random predictors "
                "get inflated AUC if labels are noisy.\n")

    print(f"\nReport written to {md_path}")


if __name__ == "__main__":
    main()
