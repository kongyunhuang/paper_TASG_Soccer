#!/usr/bin/env python3
"""
holdout_2526_dest_rules.py, destination rule baselines on the held-out season
=============================================================================
Applies the rules and parameters of dest_rule_baselines.py unchanged to the 2025/26 held-out matches.
Imports table, rule, scores, start_cell and the grids from dest_rule_baselines. The tables are built from the training
seasons only (the dest dev set of build_dataset, a sample of 100,000 passes), and a and b are chosen as in that script on
a fifth of the training-season matches held out from the table (they must equal the values in
results/dest_rule_baselines.md, a=1 with teammates only, a=2 and b=0.5 with opponents). Neither 2024/25 nor 2025/26 is
used for any choice.
  Reproduction  recomputes the five rules on the 202 clean matches of 2024/25 and compares them, to four decimals, with
                results/dest_rule_baselines.md. The held-out step runs only if all values match.
  Held-out      (with --holdout) computes the five rules on the dest task of the 2025/26 held-out set and stores the
                per-event top-1 and log probability of the true cell, for the bootstrap of holdout_2526_summarize.py.

Inputs   as dest_rule_baselines.py, plus data/cache/holdout_2526/clean_matches_2526.parquet and results/dest_rule_baselines.md
Outputs  data/cache/holdout_2526/dest_rules_repro.json
         data/cache/holdout_2526/dest_rules_holdout.npz and results/holdout_2526_dest_rules.json (with --holdout)

Usage
  PYTHONPATH=. python -u scripts/eval/holdout_2526_dest_rules.py              (reproduction only)
  PYTHONPATH=. python -u scripts/eval/holdout_2526_dest_rules.py --holdout    (reproduction and held-out set)

Last modified 2026-10-28
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from scripts.eval.dest_rule_baselines import GRID_A, GRID_B, rule, scores, start_cell, table
from scripts.eval.holdout_2526_dataset import build_holdout
from scripts.eval.summarize_u1_clean import clean_matches
from scripts.training.train_unified_clean import N_DEST, build_dataset

OUT_DIR = Path("data/cache/holdout_2526")
ORIG_MD = Path("results/dest_rule_baselines.md")
HOLD_JSON = Path("results/holdout_2526_dest_rules.json")


def all_rules(T, pm, X, sm, best_a, best_ab):
    st = start_cell(X)
    flat = lambda c: np.asarray(sm[:, c]).reshape(len(sm), -1)
    mt, ot = flat(0), flat(1)
    n = len(X)
    return {"Uniform": np.full((n, N_DEST), 1 / N_DEST), "Marginal": np.tile(pm, (n, 1)), "Start-cell table": T[st],
            f"Table times teammates (a={best_a[1]})": rule(T, st, mt, ot, best_a[1], 0),
            f"Table times teammates over opponents (a={best_ab[1]}, b={best_ab[2]})": rule(T, st, mt, ot, best_ab[1], best_ab[2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = build_dataset()["dest"]
    dev, test = d["dev"], d["test"]
    yd = dev["y"].astype(int)
    sd = start_cell(dev["X"])
    flat = lambda sm, c: np.asarray(sm[:, c]).reshape(len(sm), -1)
    md, od = flat(dev["sm"], 0), flat(dev["sm"], 1)
    matches = np.sort(np.unique(dev["match"]))
    perm = np.random.RandomState(20261002).permutation(len(matches))
    tr = np.isin(dev["match"], matches[perm[:int(0.8 * len(matches))]])
    va = ~tr
    Ttr = table(sd[tr], yd[tr])
    grid = [(scores(rule(Ttr, sd[va], md[va], od[va], a_, b_), yd[va])["logloss"], a_, b_) for a_, b_ in itertools.product(GRID_A, GRID_B)]
    best_a = min((g for g in grid if g[2] == 0), key=lambda g: g[0])
    best_ab = min(grid, key=lambda g: g[0])
    T = table(sd, yd)
    pm = np.bincount(yd, minlength=N_DEST) + 1.0
    pm = pm / pm.sum()
    print(f"[select] a={best_a[1]}  a={best_ab[1]} b={best_ab[2]}  ({time.time() - t0:.0f}s)", flush=True)

    # reproduction on 2024/25
    clean = clean_matches()
    ok = np.array([int(m) in clean for m in test["match"]])
    yt = test["y"][ok].astype(int)
    rep = {}
    orig = {}
    for line in ORIG_MD.read_text(encoding="utf8").splitlines():
        m = re.match(r"\|\s*(.+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
        if m and m.group(1) != "rule":
            orig[m.group(1)] = dict(top1=float(m.group(2)), top3=float(m.group(3)), logloss=float(m.group(4)))
    all_ok = True
    for name, P in all_rules(T, pm, test["X"][ok], test["sm"][ok], best_a, best_ab).items():
        s = scores(P, yt)
        o = orig.get(name)
        same = o is not None and all(round(s[k], 4) == o[k] for k in ("top1", "top3", "logloss"))
        all_ok &= same
        rep[name] = dict(mine={k: round(v, 4) for k, v in s.items()}, orig=o, same=bool(same))
        print(f"[repro] {name:34s} top1 {s['top1']:.4f} logloss {s['logloss']:.4f}  reference {o}  {'same' if same else 'different'}", flush=True)
    rep["_pass"] = bool(all_ok)
    rep["_n_test_202"] = int(ok.sum())
    rep["_params"] = dict(a_only=best_a[1], a=best_ab[1], b=best_ab[2])
    json.dump(rep, open(OUT_DIR / "dest_rules_repro.json", "w"), ensure_ascii=False, indent=1)
    print(f"[repro] {'all values match' if all_ok else 'some values differ'}", flush=True)
    if not all_ok or not a.holdout:
        return
    h = build_holdout(["dest"])["dest"]
    yh = h["y"].astype(int)
    res = {}
    save = dict(event_id=h["eid"], match_id=h["match"], y=yh.astype(np.int16))
    for i, (name, P) in enumerate(all_rules(T, pm, h["X"], h["sm"], best_a, best_ab).items()):
        s = scores(P, yh)
        s["skill"] = 1 - s["logloss"] / math.log(N_DEST)
        res[name] = {k: round(v, 5) for k, v in s.items()}
        save[f"rule{i}_top1"] = P.argmax(1).astype(np.int16)
        save[f"rule{i}_logp_true"] = np.log(P[np.arange(len(yh)), yh]).astype(np.float32)
        print(f"[holdout] {name:34s} top1 {s['top1']:.4f} top3 {s['top3']:.4f} logloss {s['logloss']:.4f}", flush=True)
    res["_rule_order"] = list(res.keys())
    res["_n_holdout"] = int(len(yh))
    json.dump(res, open(HOLD_JSON, "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(OUT_DIR / "dest_rules_holdout.npz", **save)
    print(f"[done] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
