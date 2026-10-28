#!/usr/bin/env python3
"""
holdout_2526_predict_unified.py, unified-model inference for the hold-out confirmation
=====================================================================================
With the saved weights, first reproduces the per-event 24/25 predictions, then (phase 2) predicts the 25/26 hold-out set.
Three steps for every {tag}_s{seed}.
  1  Load the .pt (weights and scaler) and the json of the same name (fusion, encoder, channels, rich_scalars, hard_gate,
     shuffled).
  2  Reproduction. Build the original test set with train_unified_clean.build_dataset (eight tasks, fixed subsampling),
     process it in the order of the training script (rich scalars, shuffling with RandomState(777) first for dev and then
     for test, two-channel cut, scaler), predict and compare with the original npz event by event. task, y and event_id
     must be identical, and logit_bin, dest_logp_true and gate_mean must agree element-wise within
     |diff| <= max(1e-6, 2.4e-7 x |value|), that is within two float32 ulp (differences across machines are one or two ulp,
     and at a destination log-probability of -15 one ulp is already 1.9e-6). Counts above both tolerances are recorded.
     Nothing further runs if the reproduction fails.
  3  Hold-out (only with --holdout, never in phase 1). Build the 25/26 hold-out set with holdout_2526_dataset.build_holdout
     (six tasks, all events), process it the same way and predict; save task, y, logit_bin, dest_logp_true, dest_top1,
     gate_mean, match_id and event_id. Tasks listed in shuffled are shuffled once more within the hold-out set with
     RandomState(777), as in the training intervention.
No file in u1_clean is changed. The weights directory can be set (the retrained u1 s1 and s2 are in data/cache/holdout_2526/retrain/).

Inputs   {weights_dir}/{tag}_s{seed}.pt and .json, data/cache/u1_clean/{tag}_s{seed}.npz (reference for the reproduction)
Outputs  data/cache/holdout_2526/unified/{tag}_s{seed}_repro.json      reproduction check
         data/cache/holdout_2526/unified/{tag}_s{seed}_holdout.npz     per-event hold-out predictions (with --holdout)

Usage
  PYTHONPATH=. python -u scripts/eval/holdout_2526_predict_unified.py --tag u1rich --seeds 0,1,2            (reproduction only)
  PYTHONPATH=. python -u scripts/eval/holdout_2526_predict_unified.py --tag u1 --seeds 1,2 --weights_dir data/cache/holdout_2526/retrain
  PYTHONPATH=. python -u scripts/eval/holdout_2526_predict_unified.py --tag u1rich --seeds 0,1,2 --holdout   (phase 2)

Last modified 2026-10-28
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from scripts.eval.holdout_2526_dataset import HOLDOUT_TASKS, build_holdout
from scripts.training.train_unified_clean import (N_DEST, TASK_ID, TASKS, UnifiedGatingClean, UnifiedGatingFCN, build_dataset,
                                                  predict, rich_scalars)

U1 = Path("data/cache/u1_clean")
OUT = Path("data/cache/holdout_2526/unified")
TOL = 1e-6        # absolute tolerance
REL_TOL = 2.4e-7  # relative tolerance, about two float32 ulp


def load_model(weights_dir, tag, seed, n_feat):
    ck = torch.load(weights_dir / f"{tag}_s{seed}.pt", weights_only=False)
    meta = json.load(open(weights_dir / f"{tag}_s{seed}.json"))
    Model = UnifiedGatingClean if meta.get("encoder", "pooled") == "pooled" else UnifiedGatingFCN
    ch = meta.get("channels", 7)
    if meta.get("hard_gate"):
        model = Model(n_feat, len(TASKS), fusion=meta["fusion"], cnn_channels=ch, hard_gate=True)
    else:
        model = Model(n_feat, len(TASKS), fusion=meta["fusion"], cnn_channels=ch)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, meta, ck


def prepare(data, parts, meta, rs):
    """Process in the order of the training script. Rich scalars (before shuffling), per-task shuffling of the spatial tensors (rs is
    called in the same order as in training), concatenation of the tasks, two-channel cut. parts lists the parts to concatenate
    (the original test set uses only test from ("dev","test"), but the shuffling must first consume the permutation of dev)."""
    tasks = [t for t in TASKS if t in data]
    if meta.get("rich_scalars"):
        for t in tasks:
            for part in data[t]:
                d = data[t][part]
                if "X" not in d:
                    continue  # dev is only a placeholder that holds the count
                d["X"] = np.hstack([d["X"], rich_scalars(d["sm"], d["X"][:, 0], d["X"][:, 1])])
    for t in meta.get("shuffled", []):
        if t not in data:
            continue
        for part in data[t]:
            d = data[t][part]
            n = d["_n"] if "_n" in d else len(d["sm"])
            perm = rs.permutation(n)  # as in the training script, one permutation for dev, then one for test
            if d.get("sm") is not None:
                d["sm"] = d["sm"][perm]
    out = {}
    for part in parts:
        sel = {k: np.concatenate([data[t][part][k] for t in tasks]) for k in ("X", "y", "sm", "match", "eid")}
        sel["task"] = np.concatenate([np.full(len(data[t][part]["y"]), TASK_ID[t]) for t in tasks])
        if meta.get("channels", 7) == 2:
            sel["sm"] = np.ascontiguousarray(sel["sm"][:, :2])
        out[part] = sel
    return out


def standardize(X, mean, scale):
    """Standardise in float32 in place, subtracting the mean and then dividing by the scale (one rounding each), as
    sklearn StandardScaler.transform does. Computing in float64 and casting to float32 differs by one or two ulp and does not
    reproduce to 1e-6."""
    Z = np.array(X, dtype=np.float32, copy=True)
    Z -= mean
    Z /= scale
    return Z


def run_model(model, ck, sel):
    X = torch.tensor(standardize(sel["X"], ck["scaler_mean"], ck["scaler_scale"]), dtype=torch.float32)
    S = torch.tensor(sel["sm"])
    T = torch.tensor(sel["task"], dtype=torch.long)
    tb, td, tg = predict(model, X, S, T)
    yy = sel["y"].astype(int)
    return dict(task=sel["task"].astype(np.int8), y=yy.astype(np.int16), logit_bin=tb.astype(np.float32),
                dest_logp_true=td[np.arange(len(yy)), np.clip(yy, 0, N_DEST - 1)].astype(np.float32),
                dest_top1=td.argmax(1).astype(np.int16), gate_mean=tg.astype(np.float32), match_id=sel["match"], event_id=sel["eid"])


def compare(pred, orig_path):
    z = np.load(orig_path, allow_pickle=True)
    r = dict(n=int(len(z["task"])), same_n=bool(len(z["task"]) == len(pred["task"])))
    if not r["same_n"]:
        r["pass"] = False
        return r
    r["task_equal"] = bool(np.array_equal(z["task"], pred["task"]))
    r["y_equal"] = bool(np.array_equal(z["y"], pred["y"]))
    r["eid_equal"] = bool(np.array_equal(z["event_id"].astype(str), pred["event_id"].astype(str)))
    r["top1_equal"] = bool(np.array_equal(z["dest_top1"], pred["dest_top1"]))
    for k in ("logit_bin", "dest_logp_true", "gate_mean"):
        a, b = z[k].astype(np.float64), pred[k].astype(np.float64)
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.abs(a - b)
        d[both_nan] = 0.0
        r[f"{k}_max_abs_diff"] = float(np.nanmax(d)) if len(d) else 0.0
        r[f"{k}_n_over_abs_tol"] = int(np.sum(d > TOL))
        # Relative tolerance. One float32 ulp at magnitude |a| is about 1.2e-7 x |a|; differences across machines are one or two ulp, and at a destination log-probability of -15 one ulp is 1.9e-6
        r[f"{k}_n_over_rel_tol"] = int(np.sum(d > np.maximum(TOL, REL_TOL * np.abs(np.nan_to_num(a)))))
    r["pass"] = bool(r["task_equal"] and r["y_equal"] and r["eid_equal"] and all(r[f"{k}_n_over_rel_tol"] == 0 for k in ("logit_bin", "dest_logp_true", "gate_mean")))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--weights_dir", default=str(U1))
    ap.add_argument("--holdout", action="store_true", help="phase 2 only. Predict the 25/26 hold-out set")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TASG_THREADS", "4")))
    OUT.mkdir(parents=True, exist_ok=True)
    wdir = Path(a.weights_dir)
    seeds = [int(s) for s in a.seeds.split(",")]
    t0 = time.time()
    data_orig = build_dataset()
    print(f"[data] original test set built  ({time.time() - t0:.0f}s)", flush=True)
    # dev only needs its count (consumed by the shuffling permutation); the tensors are released
    dev_n = {t: len(data_orig[t]["dev"]["y"]) for t in TASKS}
    for t in TASKS:
        data_orig[t]["dev"] = {"_n": dev_n[t], "sm": None}
    hold = build_holdout() if a.holdout else None
    for seed in seeds:
        meta = json.load(open(wdir / f"{a.tag}_s{seed}.json"))
        # Reproduce 24/25
        data = {t: {"dev": dict(data_orig[t]["dev"]), "test": {k: v.copy() for k, v in data_orig[t]["test"].items()}} for t in TASKS}
        rs = np.random.RandomState(777)
        sel = prepare(data, ("test",), meta, rs)["test"]
        model, meta, ck = load_model(wdir, a.tag, seed, sel["X"].shape[1])
        pred = run_model(model, ck, sel)
        rep = compare(pred, U1 / f"{a.tag}_s{seed}.npz")
        rep.update(tag=a.tag, seed=seed, weights_dir=str(wdir), when=time.strftime("%Y-%m-%d %H:%M"))
        json.dump(rep, open(OUT / f"{a.tag}_s{seed}_repro.json", "w"), ensure_ascii=False, indent=1)
        print(f"[repro] {a.tag} s{seed}  pass={rep['pass']}  " + " ".join(f"{k}={rep[k]:.2e}" for k in rep if k.endswith("max_abs_diff")), flush=True)
        if not rep["pass"]:
            print(f"[stop] {a.tag} s{seed} reproduction failed, hold-out skipped", flush=True)
            continue
        if a.holdout:
            hd = {t: {"hold": {k: v.copy() for k, v in hold[t].items()}} for t in HOLDOUT_TASKS}
            rs2 = np.random.RandomState(777)
            selh = prepare(hd, ("hold",), meta, rs2)["hold"]
            ph = run_model(model, ck, selh)
            np.savez_compressed(OUT / f"{a.tag}_s{seed}_holdout.npz", **ph)
            print(f"[holdout] {a.tag} s{seed}  n {len(ph['task']):,}  saved  ({time.time() - t0:.0f}s)", flush=True)
    print(f"[done] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
