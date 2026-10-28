#!/usr/bin/env python3
"""
holdout_2526_retrain_unified.py, retrained unified-model weights for the hold-out confirmation
==============================================================================================
Reruns seeds 1 and 2 of u1 with the original command and writes to holdout_2526/retrain/, leaving u1_clean untouched.
Of the reference u1 runs only seed 3 has saved weights (seed 0 is replaced by u1chk_s0). train_unified_clean.py skips a run
whose json exists, and a new tag would be picked up by summarize_u1_clean.py as a new run in the summary table. This script
therefore imports train_unified_clean, points its OUT_DIR to data/cache/holdout_2526/retrain/, and calls main() with the
original arguments (--seed N --tag u1, all others default). The training code is unchanged.
  --verify  after training, compare with data/cache/u1_clean/u1_s{seed}.json and .npz. In the json, test, val and gate_mean
            of all eight tasks, the history (per-epoch train_loss, val_score, val_gate) and best_val_score must be equal.
            In the npz, task, y and event_id must be identical per event, and logit_bin, dest_logp_true and gate_mean must
            agree element-wise within |diff| <= max(1e-6, 2.4e-7 x |value|).
            The result is written to data/cache/holdout_2526/retrain/u1_s{seed}_verify.json.

Inputs   as train_unified_clean.py
Outputs  data/cache/holdout_2526/retrain/u1_s{seed}.{json,pt,npz}, u1_s{seed}_verify.json

Usage
  PYTHONPATH=. python -u scripts/eval/holdout_2526_retrain_unified.py --seed 1 --verify
  PYTHONPATH=. python -u scripts/eval/holdout_2526_retrain_unified.py --seed 1 --verify_only   (check only, no training)

Last modified 2026-10-28
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import scripts.training.train_unified_clean as tuc

RETRAIN_DIR = Path("data/cache/holdout_2526/retrain")
U1 = Path("data/cache/u1_clean")
TOL, REL_TOL = 1e-6, 2.4e-7


def verify(seed, tag="u1"):
    ja = json.load(open(U1 / f"{tag}_s{seed}.json"))
    jb = json.load(open(RETRAIN_DIR / f"{tag}_s{seed}.json"))
    r = dict(tag=tag, seed=seed, when=time.strftime("%Y-%m-%d %H:%M"), host_orig=ja.get("host"), host_retrain=jb.get("host"))
    r["json_test_equal"] = all(ja["tasks"][t]["test"] == jb["tasks"][t]["test"] for t in ja["tasks"])
    r["json_val_equal"] = all(ja["tasks"][t]["val"] == jb["tasks"][t]["val"] for t in ja["tasks"])
    r["json_gate_equal"] = all(ja["tasks"][t]["gate_mean"] == jb["tasks"][t]["gate_mean"] for t in ja["tasks"])
    # Compare only the history fields present in both runs (the original u1 s1 and s2 histories record epoch, train_loss and val_score, without val_gate)
    hk = sorted(set(ja["history"][0]) & set(jb["history"][0])) if ja["history"] and jb["history"] else []
    r["history_fields_compared"] = hk
    r["history_fields_only_in_retrain"] = sorted(set(jb["history"][0]) - set(ja["history"][0])) if jb["history"] else []
    r["history_equal"] = len(ja["history"]) == len(jb["history"]) and all({k: x[k] for k in hk} == {k: y[k] for k in hk} for x, y in zip(ja["history"], jb["history"]))
    r["best_val_equal"] = ja["best_val_score"] == jb["best_val_score"]
    r["n_epochs_equal"] = ja["n_epochs"] == jb["n_epochs"]
    za = np.load(U1 / f"{tag}_s{seed}.npz", allow_pickle=True)
    zb = np.load(RETRAIN_DIR / f"{tag}_s{seed}.npz", allow_pickle=True)
    r["npz_task_equal"] = bool(np.array_equal(za["task"], zb["task"]))
    r["npz_y_equal"] = bool(np.array_equal(za["y"], zb["y"]))
    r["npz_eid_equal"] = bool(np.array_equal(za["event_id"].astype(str), zb["event_id"].astype(str)))
    r["npz_top1_equal"] = bool(np.array_equal(za["dest_top1"], zb["dest_top1"]))
    ok = all(r[k] for k in ("json_test_equal", "json_val_equal", "json_gate_equal", "history_equal", "best_val_equal", "npz_task_equal", "npz_y_equal", "npz_eid_equal"))
    for k in ("logit_bin", "dest_logp_true", "gate_mean"):
        a, b = za[k].astype(np.float64), zb[k].astype(np.float64)
        d = np.abs(a - b)
        d[np.isnan(a) & np.isnan(b)] = 0.0
        r[f"{k}_max_abs_diff"] = float(np.nanmax(d))
        r[f"{k}_n_over_abs_tol"] = int(np.sum(d > TOL))
        r[f"{k}_n_over_rel_tol"] = int(np.sum(d > np.maximum(TOL, REL_TOL * np.abs(np.nan_to_num(a)))))
        ok &= r[f"{k}_n_over_rel_tol"] == 0
    r["pass"] = bool(ok)
    json.dump(r, open(RETRAIN_DIR / f"{tag}_s{seed}_verify.json", "w"), ensure_ascii=False, indent=1)
    print(f"[verify] {tag} s{seed} pass={r['pass']}  json test {r['json_test_equal']} val {r['json_val_equal']} gate {r['json_gate_equal']} history {r['history_equal']}  "
          + " ".join(f"{k}={r[k]:.2e}" for k in r if k.endswith("max_abs_diff")), flush=True)
    return r["pass"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default="u1")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify_only", action="store_true")
    a = ap.parse_args()
    RETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    if not a.verify_only:
        tuc.OUT_DIR = RETRAIN_DIR  # every output path in main() is taken from this module variable
        sys.argv = ["train_unified_clean.py", "--seed", str(a.seed), "--tag", a.tag]
        print(f"[retrain] {a.tag} s{a.seed} -> {RETRAIN_DIR}  argv {sys.argv[1:]}", flush=True)
        tuc.main()
    if a.verify or a.verify_only:
        verify(a.seed, a.tag)


if __name__ == "__main__":
    main()
