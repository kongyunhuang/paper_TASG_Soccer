#!/usr/bin/env python3
"""
audit_split_integrity: Stage 10 diagnostic
============================================
Verify the train/val/test split used by U1 Unified does not leak.

Per train_unified.py:
  - TRAIN_SEASONS = 22/23 + 23/24 (EPL+LaLiga)
  - TEST_SEASONS  = 24/25 (EPL+LaLiga)
  - Within TRAIN_SEASONS: shuffle match_id (seed 42), split 80/20 -> train/val
  - test = entire TEST_SEASONS

Checks performed:
  C1. Match-id disjointness  (train ∩ val = ∅, (train ∪ val) ∩ test = ∅)
  C2. Season composition      (train+val ⊂ TRAIN_SEASONS, test ⊂ TEST_SEASONS)
  C3. Sample counts per task per split
  C4. Per-task class balance per split (pos_rate train vs val vs test)
       - Large drift = label distribution shift across split = potential bias
  C5. Event-id uniqueness within each split
  C6. Player overlap (informational only: per-match split allows player overlap
       since the same player appears in multiple matches; this is standard
       practice for football models, not a leakage source)
  C7. Date / season ordering (test seasons strictly chronologically later)
  C8. Seed determinism (rerun with seed 42 -> identical split)

Input:  data/L1_events_v3.parquet
Output: audit/split_integrity_report.md + stdout

Usage:
  PYTHONPATH=. python scripts/audit/audit_split_integrity.py

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("data")
AUDIT = Path("audit"); AUDIT.mkdir(exist_ok=True)

FULL_SEASONS = [
    "2_235_2022_23", "2_281_2023_24", "2_317_2024_25",
    "11_235_2022_23", "11_281_2023_24", "11_317_2024_25",
]
TRAIN_SEASONS = ["2_235_2022_23", "2_281_2023_24",
                 "11_235_2022_23", "11_281_2023_24"]
TEST_SEASONS = ["2_317_2024_25", "11_317_2024_25"]

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
}


def build_split(df: pd.DataFrame, seed: int = 42):
    """Replicate train_unified.py split logic exactly."""
    np.random.seed(seed)

    is_train_season = np.isin(df["season_dir"].values, TRAIN_SEASONS)
    is_test_season = np.isin(df["season_dir"].values, TEST_SEASONS)
    match_ids = df["match_id"].values

    train_matches = np.unique(match_ids[is_train_season])
    np.random.shuffle(train_matches)
    n_t = int(0.8 * len(train_matches))
    train_ids = train_matches[:n_t]
    val_ids = train_matches[n_t:]

    train_mask = np.isin(match_ids, train_ids)
    val_mask = np.isin(match_ids, val_ids)
    test_mask = is_test_season

    return train_mask, val_mask, test_mask, train_ids, val_ids


def main():
    print("=" * 78)
    print("Stage 10 audit: train/val/test split integrity (U1 Unified)")
    print("=" * 78)

    df = pd.read_parquet(CACHE / "L1_events_v3.parquet")
    df = df[df["season_dir"].isin(FULL_SEASONS)].reset_index(drop=True)
    print(f"Loaded {len(df):,} rows.\n")

    train_mask, val_mask, test_mask, train_ids, val_ids = build_split(df, seed=42)
    test_ids = np.unique(df["match_id"].values[test_mask])

    results = {}

    # ── C1. match-id disjointness ──────────────────────────────────────
    inter_tv = np.intersect1d(train_ids, val_ids)
    inter_tt = np.intersect1d(np.union1d(train_ids, val_ids), test_ids)
    c1_ok = (len(inter_tv) == 0) and (len(inter_tt) == 0)
    print(f"[C1] Match-id disjointness")
    print(f"     train ∩ val matches:      {len(inter_tv)}  ({'OK' if len(inter_tv)==0 else 'FAIL'})")
    print(f"     (train+val) ∩ test:       {len(inter_tt)}  ({'OK' if len(inter_tt)==0 else 'FAIL'})")
    print(f"     verdict: {'OK' if c1_ok else 'FAIL'}\n")
    results["C1_match_disjoint"] = c1_ok

    # ── C2. season composition ─────────────────────────────────────────
    seasons_arr = df["season_dir"].values
    train_seasons = set(seasons_arr[train_mask])
    val_seasons = set(seasons_arr[val_mask])
    test_seasons = set(seasons_arr[test_mask])
    c2_ok = (train_seasons.issubset(set(TRAIN_SEASONS))
             and val_seasons.issubset(set(TRAIN_SEASONS))
             and test_seasons.issubset(set(TEST_SEASONS)))
    print(f"[C2] Season composition")
    print(f"     train seasons:  {sorted(train_seasons)}")
    print(f"     val seasons:    {sorted(val_seasons)}")
    print(f"     test seasons:   {sorted(test_seasons)}")
    print(f"     verdict: {'OK' if c2_ok else 'FAIL'}\n")
    results["C2_season_composition"] = c2_ok

    # ── C3. sample counts per task per split ───────────────────────────
    print(f"[C3] Sample counts per task per split")
    print(f"     {'task':14s}  {'train':>9s}  {'val':>9s}  {'test':>9s}")
    task_counts = {}
    for task_name, cfg in BINARY_TASKS.items():
        df_task = df[df["type_name"] == cfg["type_name"]].copy()
        if cfg["filter_fn"]:
            df_task = cfg["filter_fn"](df_task)
        task_idx = df_task.index.values
        n_tr = int(train_mask[task_idx].sum())
        n_va = int(val_mask[task_idx].sum())
        n_te = int(test_mask[task_idx].sum())
        task_counts[task_name] = {"train": n_tr, "val": n_va, "test": n_te}
        print(f"     {task_name:14s}  {n_tr:>9,}  {n_va:>9,}  {n_te:>9,}")
    results["C3_task_counts"] = task_counts

    # Pressure separately (label set requires full df context)
    n_pressure = int((df["type_name"] == "Pressure").sum())
    pr_idx = df.index[df["type_name"] == "Pressure"].values
    print(f"     {'pressure':14s}  "
          f"{int(train_mask[pr_idx].sum()):>9,}  "
          f"{int(val_mask[pr_idx].sum()):>9,}  "
          f"{int(test_mask[pr_idx].sum()):>9,}")
    print()

    # ── C4. class balance per task per split ───────────────────────────
    print(f"[C4] Per-task pos_rate per split  (drift > 0.05 might bias evaluation)")
    print(f"     {'task':14s}  {'train':>9s}  {'val':>9s}  {'test':>9s}  "
          f"{'|tr-va|':>8s}  {'|tr-te|':>8s}")
    balance = {}
    for task_name, cfg in BINARY_TASKS.items():
        df_task = df[df["type_name"] == cfg["type_name"]].copy()
        if cfg["filter_fn"]:
            df_task = cfg["filter_fn"](df_task)
        labels = cfg["label_fn"](df_task)
        idx = df_task.index.values
        tr_lab = labels[train_mask[idx]]
        va_lab = labels[val_mask[idx]]
        te_lab = labels[test_mask[idx]]
        pr_tr = float(tr_lab.mean()) if len(tr_lab) else float("nan")
        pr_va = float(va_lab.mean()) if len(va_lab) else float("nan")
        pr_te = float(te_lab.mean()) if len(te_lab) else float("nan")
        d_tv = abs(pr_tr - pr_va) if not np.isnan(pr_tr + pr_va) else float("nan")
        d_tt = abs(pr_tr - pr_te) if not np.isnan(pr_tr + pr_te) else float("nan")
        balance[task_name] = {"train": pr_tr, "val": pr_va, "test": pr_te,
                              "drift_tr_va": d_tv, "drift_tr_te": d_tt}
        flag = " <-- DRIFT" if d_tt > 0.05 else ""
        print(f"     {task_name:14s}  "
              f"{pr_tr:>9.4f}  {pr_va:>9.4f}  {pr_te:>9.4f}  "
              f"{d_tv:>8.4f}  {d_tt:>8.4f}{flag}")
    results["C4_class_balance"] = balance
    print()

    # ── C5. event-id uniqueness within each split ──────────────────────
    print(f"[C5] Event-id uniqueness within each split")
    eids = df["event_id"].values
    eid_tr_uniq = len(np.unique(eids[train_mask])) == train_mask.sum()
    eid_va_uniq = len(np.unique(eids[val_mask])) == val_mask.sum()
    eid_te_uniq = len(np.unique(eids[test_mask])) == test_mask.sum()
    print(f"     train unique event_ids: {eid_tr_uniq}")
    print(f"     val   unique event_ids: {eid_va_uniq}")
    print(f"     test  unique event_ids: {eid_te_uniq}")
    c5_ok = eid_tr_uniq and eid_va_uniq and eid_te_uniq
    # cross-split event_id overlap
    inter_eids_tv = np.intersect1d(eids[train_mask], eids[val_mask])
    inter_eids_tt = np.intersect1d(eids[train_mask], eids[test_mask])
    print(f"     train ∩ val event_ids: {len(inter_eids_tv)}")
    print(f"     train ∩ test event_ids: {len(inter_eids_tt)}")
    c5_ok = c5_ok and len(inter_eids_tv) == 0 and len(inter_eids_tt) == 0
    print(f"     verdict: {'OK' if c5_ok else 'FAIL'}\n")
    results["C5_event_id_unique"] = c5_ok

    # ── C6. player overlap (informational) ─────────────────────────────
    if "player_id" in df.columns:
        p_tr = set(df["player_id"].values[train_mask])
        p_va = set(df["player_id"].values[val_mask])
        p_te = set(df["player_id"].values[test_mask])
        p_tr.discard(None)
        p_va.discard(None)
        p_te.discard(None)
        overlap_tv = len(p_tr & p_va)
        overlap_tt = len(p_tr & p_te)
        union_train_val = p_tr | p_va
        print(f"[C6] Player overlap (informational: per-match split intentionally"
              " allows player overlap)")
        print(f"     unique players: train={len(p_tr)}, val={len(p_va)}, "
              f"test={len(p_te)}")
        print(f"     train ∩ val players:    {overlap_tv}  "
              f"({100*overlap_tv/max(1,len(p_va)):.1f}% of val players)")
        print(f"     train ∩ test players:   {overlap_tt}  "
              f"({100*overlap_tt/max(1,len(p_te)):.1f}% of test players)")
        print(f"     train+val ∩ test:       {len(union_train_val & p_te)}  "
              f"({100*len(union_train_val & p_te)/max(1,len(p_te)):.1f}% of test players)")
        print(f"     verdict: INFO (this is intended; player-aware tasks may need stricter split)\n")
        results["C6_player_overlap"] = {
            "n_train_players": len(p_tr),
            "n_val_players": len(p_va),
            "n_test_players": len(p_te),
            "overlap_train_val": overlap_tv,
            "overlap_train_test": overlap_tt,
            "overlap_unionTV_test": len(union_train_val & p_te),
        }

    # ── C7. season chronology ──────────────────────────────────────────
    # season strings encode year: e.g. 2_235_2022_23 -> 2022/23
    def season_year(s):
        # take last 7 chars "_YYYY_YY"
        parts = s.split("_")
        return int(parts[-2])
    tr_year_max = max(season_year(s) for s in TRAIN_SEASONS)
    te_year_min = min(season_year(s) for s in TEST_SEASONS)
    c7_ok = te_year_min > tr_year_max
    print(f"[C7] Season chronology")
    print(f"     max train season year: {tr_year_max}")
    print(f"     min test  season year: {te_year_min}")
    print(f"     verdict: {'OK (test strictly after train)' if c7_ok else 'FAIL'}\n")
    results["C7_chronology"] = c7_ok

    # ── C8. seed determinism ───────────────────────────────────────────
    tr2, va2, te2, _, _ = build_split(df, seed=42)
    c8_ok = (np.array_equal(tr2, train_mask) and
             np.array_equal(va2, val_mask) and
             np.array_equal(te2, test_mask))
    print(f"[C8] Seed 42 determinism (re-run produces identical split)")
    print(f"     verdict: {'OK' if c8_ok else 'FAIL'}\n")
    results["C8_deterministic"] = c8_ok

    # ── Final summary ──────────────────────────────────────────────────
    print("=" * 78)
    print("FINAL VERDICT")
    print("=" * 78)
    all_pass = all([
        results["C1_match_disjoint"],
        results["C2_season_composition"],
        results["C5_event_id_unique"],
        results["C7_chronology"],
        results["C8_deterministic"],
    ])
    print(f"  C1 match disjoint:        {'PASS' if results['C1_match_disjoint'] else 'FAIL'}")
    print(f"  C2 season composition:    {'PASS' if results['C2_season_composition'] else 'FAIL'}")
    print(f"  C5 event-id uniqueness:   {'PASS' if results['C5_event_id_unique'] else 'FAIL'}")
    print(f"  C7 chronology:            {'PASS' if results['C7_chronology'] else 'FAIL'}")
    print(f"  C8 determinism:           {'PASS' if results['C8_deterministic'] else 'FAIL'}")
    print(f"\n  Overall: {'PASS: no split leakage' if all_pass else 'FAIL: investigate'}")

    # ── Markdown report ────────────────────────────────────────────────
    md_path = AUDIT / "split_integrity_report.md"
    with open(md_path, "w") as f:
        f.write("# Train / Val / Test Split Integrity Audit: U1 Unified\n\n")
        f.write("**Date**: 2026-05-11  \n")
        f.write("**Script**: `scripts/audit/audit_split_integrity.py`  \n")
        f.write(f"**Split strategy**: per-match (seed 42); train+val from TRAIN_SEASONS, test from TEST_SEASONS.\n\n")

        f.write("## Checks\n\n")
        f.write("| ID | Check | Result |\n|---|---|---|\n")
        f.write(f"| C1 | Match-id disjoint (train ∩ val, (train+val) ∩ test) | "
                f"{'PASS' if results['C1_match_disjoint'] else 'FAIL'} |\n")
        f.write(f"| C2 | Season composition (train+val ⊂ TRAIN_SEASONS, test ⊂ TEST_SEASONS) | "
                f"{'PASS' if results['C2_season_composition'] else 'FAIL'} |\n")
        f.write(f"| C5 | Event-id uniqueness within and across splits | "
                f"{'PASS' if results['C5_event_id_unique'] else 'FAIL'} |\n")
        f.write(f"| C7 | Test seasons strictly chronologically after train | "
                f"{'PASS' if results['C7_chronology'] else 'FAIL'} |\n")
        f.write(f"| C8 | Seed 42 produces deterministic split | "
                f"{'PASS' if results['C8_deterministic'] else 'FAIL'} |\n\n")

        f.write("## C3: Sample counts per task per split\n\n")
        f.write("| Task | Train | Val | Test |\n|---|---:|---:|---:|\n")
        for tn, c in task_counts.items():
            f.write(f"| {tn} | {c['train']:,} | {c['val']:,} | {c['test']:,} |\n")
        f.write("\n")

        f.write("## C4: Class balance per split  (label pos_rate)\n\n")
        f.write("| Task | Train | Val | Test | |train-val| | |train-test| | Note |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for tn, b in balance.items():
            note = "drift > 0.05 (covariate shift?)" if b["drift_tr_te"] > 0.05 else ""
            f.write(f"| {tn} | {b['train']:.4f} | {b['val']:.4f} | {b['test']:.4f} "
                    f"| {b['drift_tr_va']:.4f} | {b['drift_tr_te']:.4f} | {note} |\n")
        f.write("\n")

        if "C6_player_overlap" in results:
            p = results["C6_player_overlap"]
            f.write("## C6: Player overlap (informational)\n\n")
            f.write(f"Per-match split intentionally allows player overlap; the same "
                    f"player appears in multiple matches across seasons.\n\n")
            f.write(f"- Unique players: train={p['n_train_players']}, "
                    f"val={p['n_val_players']}, test={p['n_test_players']}\n")
            f.write(f"- train ∩ val players: {p['overlap_train_val']}\n")
            f.write(f"- train ∩ test players: {p['overlap_train_test']}\n")
            f.write(f"- (train ∪ val) ∩ test players: {p['overlap_unionTV_test']}\n\n")
            f.write("**Interpretation**: a high player overlap between train and test is "
                    "expected because the same Premier League / La Liga roster plays across "
                    "22/23, 23/24, 24/25. For event-outcome prediction (which is per-action, "
                    "not per-player), this is standard practice and not a leakage source. "
                    "It would be a concern only for player-rating tasks.\n\n")

        f.write("## Overall verdict\n\n")
        f.write(f"{'**PASS**  (no split leakage detected).' if all_pass else '**FAIL**  (see failing checks above).'}\n")

    print(f"\nReport written to {md_path}")


if __name__ == "__main__":
    main()
