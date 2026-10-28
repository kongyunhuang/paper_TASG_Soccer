#!/usr/bin/env python3
"""
holdout_2526_dataset.py, 25/26 hold-out data built exactly as in train_unified_clean
=====================================================================================
train_unified_clean.build_dataset is fixed to FULL_SEASONS and the 24/25 state table, so it cannot produce 25/26.
build_holdout copies its feature columns, label definitions, tensor lookup and filters line by line, but takes the seasons
and matches as arguments and does not subsample (the whole hold-out set is used).
The hold-out tasks are pass, dest, shot, interception, ball_recovery and pressure. The re-oriented dribble and tackle tensors
do not cover 25/26, so those two tasks are not built here.
  --verify  check on the 202 clean matches of 24/25. The original build_dataset builds its test set (fixed subsampling),
            build_holdout builds the same matches, and after alignment by event_id X, y, sm and match must be identical
            element by element, with every event_id of the original test set present. The 25/26 set is used only if
            this check passes.
  default   print the size and label mean of every 25/26 hold-out task without saving. Tensors are read with mmap.
Nothing is written to u1_clean, single_clean or single_clean_rich.

Inputs   as train_unified_clean.py, plus data/cache/holdout_2526/clean_matches_2526.parquet
Outputs  results/holdout_2526_dataset_verify.md (with --verify)

Usage
  PYTHONPATH=. python -u scripts/eval/holdout_2526_dataset.py --verify
  PYTHONPATH=. python -u scripts/eval/holdout_2526_dataset.py

Last modified 2026-10-28
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.training.train_all import CACHE_DIR, TEST_SEASONS
from scripts.training.train_unified_clean import F_BASE, F_PASS, F_SB, pressure_labels_5s

HOLDOUT_SEASONS = ["11_318_2025_26_part", "2_318_2025_26_part"]
HOLDOUT_TASKS = ["pass", "dest", "shot", "interception", "ball_recovery", "pressure"]
HOLDOUT_DIR = CACHE_DIR / "holdout_2526"
VERIFY_MD = Path("results/holdout_2526_dataset_verify.md")
SRC = {"pass": "pass", "dest": "pass", "shot": "shot", "interception": "interception", "ball_recovery": "ball_recovery", "pressure": "pressure"}
TYPE_NAME = {"pass": "Pass", "shot": "Shot", "interception": "Interception", "ball_recovery": "Ball Recovery", "pressure": "Pressure"}


def holdout_match_ids():
    return set(pd.read_parquet(HOLDOUT_DIR / "clean_matches_2526.parquet")["match_id"].astype(int))


def load_events(seasons, matches=None):
    """Read the event table with the same preprocessing as build_dataset (plus a match filter). The table keeps all event types of these matches, as the 5 s labels need them."""
    cols = ["event_id", "match_id", "period", "timestamp", "team_id", "type_name", "season_dir"] + F_BASE + F_PASS + F_SB + \
           ["pass_end_location_x", "pass_end_location_y", "pass_outcome_name", "shot_outcome_name", "shot_type_name",
            "interception_outcome_name", "ball_recovery_failure"]
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet", columns=cols)
    df = df[df["season_dir"].isin(seasons)]
    if matches is not None:
        df = df[df["match_id"].astype(int).isin(matches)]
    df = df.reset_index(drop=True)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    for c in ["pass_is_progressive", "pass_cross", "pass_switch", "pass_through_ball", "pass_cut_back"]:
        df[c] = df[c].fillna(False).astype(int)
    return df


def build_task(df, press_y, task):
    """Full hold-out set of one task. Filters, labels and features are identical to build_dataset line by line; no subsampling."""
    zeros_pass = np.zeros(len(F_PASS), dtype=np.float32)
    src = SRC[task]
    d = df[df["type_name"] == TYPE_NAME[src]]
    if task == "shot":
        d = d[d["shot_type_name"] != "Penalty"]
    if task == "dest":
        d = d[d["pass_end_location_x"].notna() & d["pass_end_location_y"].notna()]
    sidx = pd.read_parquet(CACHE_DIR / f"action_soccermaps_{src}_idx.parquet")
    lookup = {e: i for i, e in enumerate(sidx["event_id"].values)}
    d = d[d["event_id"].isin(lookup)].reset_index(drop=True)
    smaps = np.load(CACHE_DIR / f"action_soccermaps_{src}.npy", mmap_mode="r")
    base = d[F_BASE].fillna(0).values.astype(np.float32)
    ps = d[F_PASS].fillna(0).values.astype(np.float32) if task == "pass" else np.tile(zeros_pass, (len(d), 1))
    sb = d[F_SB].fillna(0).values.astype(np.float32)
    X = np.hstack([base, ps, sb])
    if task == "pass":
        y = d["pass_outcome_name"].isna().astype(np.int64).values
    elif task == "dest":
        gx = np.clip((d["pass_end_location_x"].values / 10).astype(int), 0, 11)
        gy = np.clip((d["pass_end_location_y"].values / 10).astype(int), 0, 7)
        y = (gy * 12 + gx).astype(np.int64)
    elif task == "shot":
        y = (d["shot_outcome_name"] == "Goal").astype(np.int64).values
    elif task == "interception":
        y = d["interception_outcome_name"].isin({"Won", "Success In Play"}).astype(np.int64).values
    elif task == "ball_recovery":
        y = (d["ball_recovery_failure"] != True).astype(np.int64).values
    else:
        y = press_y.reindex(d["event_id"].values).fillna(0).values.astype(np.int64)
    rows = np.array([lookup[e] for e in d["event_id"].values])
    order = np.argsort(rows)
    sm = np.empty((len(rows), 7, 8, 12), dtype=np.float32)
    sm[order] = np.asarray(smaps[rows[order]])
    return dict(X=X, y=y, sm=sm, match=d["match_id"].values.astype(np.int64), eid=d["event_id"].values.astype(str))


def build_holdout(tasks=HOLDOUT_TASKS, seasons=HOLDOUT_SEASONS, matches=None, verbose=True):
    """Return {task: dict(X, y, sm, match, eid)}; by default all 25/26 matches that pass the proxy criteria, with all their events."""
    t0 = time.time()
    if matches is None and seasons == HOLDOUT_SEASONS:
        matches = holdout_match_ids()
    df = load_events(seasons, matches)
    press_y = pressure_labels_5s(df)
    out = {}
    for task in tasks:
        out[task] = build_task(df, press_y, task)
        if verbose:
            print(f"[holdout] {task:14s} n {len(out[task]['y']):7,}  matches {len(np.unique(out[task]['match'])):4d}  "
                  f"mean label {out[task]['y'].mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def verify():
    """Check on the 202 clean matches of 24/25 that build_holdout and the build_dataset test set are identical element by element."""
    from scripts.eval.summarize_u1_clean import clean_matches
    from scripts.training.train_unified_clean import build_dataset
    t0 = time.time()
    clean = clean_matches()
    orig = build_dataset()
    for t in list(orig):
        orig[t].pop("dev", None)  # keep only the test set, to save memory
    print(f"[verify] original build_dataset test set built  ({time.time() - t0:.0f}s)", flush=True)
    L = ["# Hold-out data construction check (202 clean matches of 24/25)", "",
         f"Generated by `scripts/eval/holdout_2526_dataset.py --verify`, {time.strftime('%Y-%m-%d %H:%M')}.",
         "The test set of the original build_dataset (fixed subsampling), restricted to the 202 matches, is aligned by event_id with the full build_holdout construction on the same matches and compared element by element.", "",
         "| Task | Original test events in the 202 matches | Events in this construction | All original event_ids present | X identical | y identical | Tensors identical | match identical | Result |", "|---|---:|---:|---|---|---|---|---|---|"]
    all_ok = True
    df = load_events(TEST_SEASONS, clean)
    press_y = pressure_labels_5s(df)
    for t in HOLDOUT_TASKS:
        o = orig[t]["test"]
        keep = np.array([int(m) in clean for m in o["match"]])
        o_eid = o["eid"][keep]
        m = build_task(df, press_y, t)  # built task by task, to save memory
        pos = pd.Series(np.arange(len(m["eid"])), index=m["eid"])
        subset = bool(pd.Index(o_eid).isin(pos.index).all())
        if not subset:
            all_ok = False
            L.append(f"| {t} | {int(keep.sum()):,} | {len(m['y']):,} | no | not compared | not compared | not compared | not compared | **failed** |")
            continue
        j = pos.loc[o_eid].values
        x_ok = bool(np.array_equal(o["X"][keep], m["X"][j]))
        y_ok = bool(np.array_equal(o["y"][keep], m["y"][j]))
        sm_ok = bool(np.array_equal(o["sm"][keep], m["sm"][j]))
        mt_ok = bool(np.array_equal(o["match"][keep], m["match"][j]))
        ok = x_ok and y_ok and sm_ok and mt_ok
        all_ok &= ok
        L.append(f"| {t} | {int(keep.sum()):,} | {len(m['y']):,} | yes | {'yes' if x_ok else 'no'} | {'yes' if y_ok else 'no'} | {'yes' if sm_ok else 'no'} | {'yes' if mt_ok else 'no'} | {'passed' if ok else '**failed**'} |")
        print(f"[verify] {t:14s} orig {int(keep.sum()):7,} mine {len(m['y']):7,} subset {subset} X {x_ok} y {y_ok} sm {sm_ok} match {mt_ok}", flush=True)
        del m, orig[t]
    L += ["", f"Overall result  {'all checks pass' if all_ok else 'some checks fail; the construction must not be used for 25/26'}.", ""]
    VERIFY_MD.write_text("\n".join(L), encoding="utf8")
    print(f"[verify] {'all checks pass' if all_ok else 'some checks fail'}  {VERIFY_MD}  ({time.time() - t0:.0f}s)", flush=True)
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        verify()
    else:
        build_holdout()
