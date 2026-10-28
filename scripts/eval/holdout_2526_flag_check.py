#!/usr/bin/env python3
"""
holdout_2526_flag_check.py, direct check of the teammate flags in the hold-out data
===================================================================================
Compares, event by event, the teammate and opponent counts of the cached tensors with the provider's visible_player_counts.
Channel 0 of the cached tensors adds 1 per teammate and channel 1 adds 1 per opponent, so the channel sums are the numbers
of points flagged teammate and opponent in the frame. sb_visible_teammates and sb_visible_opponents in the event table are
the provider's visible player counts per team (the vpc rule of results/raw_360_check_tables.md compares the frame's
teammate count with them). Classes
  agree     sum of channel 0 equals sb_visible_teammates and sum of channel 1 equals sb_visible_opponents
  reversed  sum of channel 0 equals sb_visible_opponents and sum of channel 1 equals sb_visible_teammates, and the two differ
  tie       the two sb counts are equal (undecidable)
  other     none of the above (events with a missing provider count form a class of their own)
Summarised by task and season group (22/23, 23/24, the 202 clean matches of 24/25, the other 558 matches of 24/25, 25/26);
for 25/26 the reversed share is also listed match by match.
This directly tests whether the teammate flags of the five tasks are reversed in the 146 hold-out matches. Mirrored
coordinates are at 0% for these five event types in every state (tables T4a and T4b of results/raw_360_check_tables.md),
so they are not checked again. No model is involved.

Inputs   data/cache/L1_events_v3.parquet, data/cache/action_soccermaps_{task}.npy and _idx.parquet, data/raw_scan/match_state_2425.parquet
Outputs  results/holdout_2526_flag_check.md, data/cache/holdout_2526/flag_check_by_match_2526.parquet

Usage
  PYTHONPATH=. python -u scripts/eval/holdout_2526_flag_check.py

Last modified 2026-10-28
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.eval.summarize_u1_clean import clean_matches

CACHE = Path("data/cache")
OUT_MD = Path("results/holdout_2526_flag_check.md")
OUT_PQ = CACHE / "holdout_2526" / "flag_check_by_match_2526.parquet"
TASKS = {"pass": "Pass", "shot": "Shot", "interception": "Interception", "ball_recovery": "Ball Recovery", "pressure": "Pressure"}
ORDER = ["22/23", "23/24", "24/25 clean 202", "24/25 other 558", "25/26"]
CHUNK = 100000


def grp_of(df, clean):
    s = df["season_dir"].astype(str)
    g = np.where(s.str.contains("2022_23"), "22/23", np.where(s.str.contains("2023_24"), "23/24", np.where(s.str.contains("2025_26"), "25/26", "")))
    g = pd.Series(g, index=df.index)
    m = s.str.contains("2024_25")
    g[m & df["match_id"].isin(clean)] = "24/25 clean 202"
    g[m & ~df["match_id"].isin(clean)] = "24/25 other 558"
    return g


def main():
    t0 = time.time()
    clean = clean_matches()
    df = pd.read_parquet(CACHE / "L1_events_v3.parquet", columns=["event_id", "match_id", "season_dir", "type_name", "sb_visible_teammates", "sb_visible_opponents"])
    df["match_id"] = df["match_id"].astype(int)
    df["grp"] = grp_of(df, clean)
    rows, per_match = [], []
    for task, tn in TASKS.items():
        sidx = pd.read_parquet(CACHE / f"action_soccermaps_{task}_idx.parquet")
        pos = pd.Series(np.arange(len(sidx)), index=sidx["event_id"].values)
        d = df[(df["type_name"] == tn) & df["event_id"].isin(pos.index)].copy()
        d["_row"] = pos.loc[d["event_id"].values].values
        d = d.sort_values("_row").reset_index(drop=True)
        smaps = np.load(CACHE / f"action_soccermaps_{task}.npy", mmap_mode="r")
        n_t = np.empty(len(d), dtype=np.float32)
        n_o = np.empty(len(d), dtype=np.float32)
        r = d["_row"].values
        for i in range(0, len(d), CHUNK):
            blk = np.asarray(smaps[r[i:i + CHUNK], :2])
            n_t[i:i + CHUNK] = blk[:, 0].sum((1, 2))
            n_o[i:i + CHUNK] = blk[:, 1].sum((1, 2))
        vt, vo = d["sb_visible_teammates"].values, d["sb_visible_opponents"].values
        have = ~np.isnan(vt) & ~np.isnan(vo)
        tie = have & (vt == vo)
        agree = have & ~tie & (n_t == vt) & (n_o == vo)
        rev = have & ~tie & (n_t == vo) & (n_o == vt)
        other = have & ~tie & ~agree & ~rev
        d["cls"] = np.where(~have, "provider count missing", np.where(tie, "tie", np.where(agree, "agree", np.where(rev, "reversed", "other"))))
        for g, h in d.groupby("grp"):
            vc = h["cls"].value_counts()
            n = len(h)
            dec = int(vc.get("agree", 0) + vc.get("reversed", 0))
            rows.append({"Task": task, "Season": g, "Events with frame": n, "Agree %": round(100 * vc.get("agree", 0) / n, 2), "Reversed %": round(100 * vc.get("reversed", 0) / n, 3),
                         "Tie %": round(100 * vc.get("tie", 0) / n, 1), "Other %": round(100 * vc.get("other", 0) / n, 2), "Provider count missing %": round(100 * vc.get("provider count missing", 0) / n, 2),
                         "Reversed % of decidable": round(100 * vc.get("reversed", 0) / dec, 3) if dec else float("nan")})
        h = d[d["grp"] == "25/26"]
        for mid, hh in h.groupby("match_id"):
            vc = hh["cls"].value_counts()
            dec = int(vc.get("agree", 0) + vc.get("reversed", 0))
            per_match.append({"match_id": mid, "task": task, "n": len(hh), "n_decidable": dec, "n_rev": int(vc.get("reversed", 0)), "n_other": int(vc.get("other", 0)),
                              "rev_pct_decidable": 100 * vc.get("reversed", 0) / dec if dec else float("nan")})
        print(f"[{task}] {len(d):,} events  ({time.time() - t0:.0f}s)", flush=True)
    tab = pd.DataFrame(rows)
    key = list(zip(tab["Task"], tab["Season"]))
    if len(set(key)) != len(key):
        raise RuntimeError("duplicate key in the summary table")
    tab["Season"] = pd.Categorical(tab["Season"], ORDER, ordered=True)
    tab = tab.sort_values(["Task", "Season"])
    pm = pd.DataFrame(per_match)
    if pm.duplicated(["match_id", "task"]).any():
        raise RuntimeError("duplicate key in the per-match table")
    OUT_PQ.parent.mkdir(parents=True, exist_ok=True)
    pm.to_parquet(OUT_PQ, index=False)
    piv = pm.pivot(index="match_id", columns="task", values="rev_pct_decidable")
    worst = pm.groupby("match_id").agg(n_decidable=("n_decidable", "sum"), n_rev=("n_rev", "sum"), n_other=("n_other", "sum")).reset_index()
    worst["rev_pct_all5"] = 100 * worst["n_rev"] / worst["n_decidable"]
    L = ["# Hold-out confirmation, direct check of the teammate flags of the five 25/26 tasks", "",
         f"Generated by `scripts/eval/holdout_2526_flag_check.py`, {time.strftime('%Y-%m-%d %H:%M')}. Sums of tensor channels 0 and 1 against the provider's visible_player_counts, classified event by event; no model is involved.", "",
         "## By task and season group", "", tab.to_markdown(index=False), "",
         "## 25/26 by match (five tasks combined, reversed share among decidable events)", "",
         f"Over the 146 matches, the five-task reversed share is at most {worst['rev_pct_all5'].max():.3f}% with median {worst['rev_pct_all5'].median():.3f}%; reversed events {int(worst['n_rev'].sum())} of {int(worst['n_decidable'].sum()):,} decidable; \"other\" events {int(worst['n_other'].sum())}.", "",
         worst.round(3).to_markdown(index=False), "", "## 25/26 by match and task (reversed % of decidable)", "", piv.round(2).reset_index().to_markdown(index=False), ""]
    OUT_MD.write_text("\n".join(L), encoding="utf8")
    print(tab.to_string(index=False))
    print(f"[done] {OUT_MD}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
