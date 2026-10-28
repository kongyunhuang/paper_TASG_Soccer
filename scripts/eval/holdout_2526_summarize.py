#!/usr/bin/env python3
"""
holdout_2526_summarize.py, summary and verdicts of the held-out confirmation on 2025/26
========================================================================================
Reads the per-event predictions on the held-out set, computes the metrics and match bootstrap intervals, and applies the
thresholds fixed before the run mechanically.
The threshold dictionary RULES is printed at run time and written into the report; the same thresholds are listed in
results/decision_rules.md. Grouping and pairing keys (run, seed, task, model) that occur twice raise an error instead of
being overwritten.
  --check_2425  reads the original 2024/25 predictions instead of the held-out set (u1_clean npz files and
                single_clean{_rich}/preds), restricted to the 202 clean matches, and computes the same metrics to check
                this script against results/u1_clean_summary.md. No verdicts and no bootstrap. Also stores the 2024/25
                means of every model in data/cache/holdout_2526/ref_2425_metrics.json for the held-out report.
  default       reads the outputs of the held-out inference under data/cache/holdout_2526/, computes the metrics, the
                match bootstrap (146 matches resampled --n_boot times, RandomState 20261003) and the verdicts.

Inputs   data/cache/holdout_2526/unified/{tag}_s{seed}_holdout.npz
         data/cache/holdout_2526/single{_rich}/holdout_preds/{task}_{model}_s{seed}.npz and {task}_holdout_meta.npz
         data/cache/holdout_2526/dest_rules_holdout.npz and results/holdout_2526_dest_rules.json
         data/cache/holdout_2526/match_table_2526.parquet
Outputs  results/holdout_2526_summary.md and results/holdout_2526_summary.json
         results/holdout_2526_summary_check2425.md and data/cache/holdout_2526/ref_2425_metrics.json (with --check_2425)

Usage
  PYTHONPATH=. python -u scripts/eval/holdout_2526_summarize.py --check_2425
  PYTHONPATH=. python -u scripts/eval/holdout_2526_summarize.py --n_boot 2000

Last modified 2026-10-28
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.eval.summarize_u1_clean import clean_matches

HOLD = Path("data/cache/holdout_2526")
U1 = Path("data/cache/u1_clean")
OUT_MD = Path("results/holdout_2526_summary.md")
SUMMARY_JSON = Path("results/holdout_2526_summary.json")
RULES_JSON = Path("results/holdout_2526_dest_rules.json")
REF_JSON = HOLD / "ref_2425_metrics.json"
TASKS8 = ["pass", "dest", "shot", "interception", "ball_recovery", "pressure", "dribble", "tackle"]
TASK_ID = {t: i for i, t in enumerate(TASKS8)}
TASKS6 = ["pass", "dest", "shot", "interception", "ball_recovery", "pressure"]
OUTCOME5 = ["pass", "shot", "interception", "ball_recovery", "pressure"]
UNIFIED_TAGS = {"u1": [0, 1, 2, 3], "u1rich": [0, 1, 2], "u1fcn": [0, 1, 2], "shuf_dest": [0, 1, 2], "u1concat": [0, 1, 2]}
SINGLE = {"single": {"B2_XGB_360": [0, 1, 2, 3, 4], "M2_MLP_360": [0, 1, 2, 3, 4], "M4_CNN_Full": [0, 1, 2, 3, 4], "G1_Gating": [0, 1, 2, 3, 4]},
          "single_rich": {"B2_XGB_360": [0, 1, 2, 3, 4], "M2_MLP_360": [0, 1, 2], "M4_CNN_Full": [0, 1, 2], "G1_Gating": [0, 1, 2]}}
BOOT_SEED = 20261003
C2_KEYS = ["thin_gain", "rich_gain", "xgbrich_minus_m4thin", "m4rich_minus_xgbrich", "m4thin_minus_xgbthin", "xgbrich_minus_xgbthin"]

# Thresholds fixed before the held-out run; they must not be changed
RULES = {
    "C1": dict(fcn_minus_rule_confirm=0.030, fcn_minus_m2rich_confirm=0.060, fcn_minus_rule_fail=0.010, fcn_minus_m2rich_fail=0.020),
    "C2": dict(mean_tasks=["pass", "ball_recovery", "pressure"], xgbrich_minus_m4thin_confirm=-0.003, rich_gain_ratio_confirm=0.5, rich_gain_abs_confirm=0.010,
               pressure_thin_gain_min=0.010, pressure_rich_gain_max=0.010,
               xgbrich_minus_m4thin_fail=-0.008, pressure_rich_over_thin_fail=0.5),
    "C3": dict(mean_tasks=["pass", "ball_recovery", "pressure"], pass_abs_confirm=0.006, pressure_abs_confirm=0.006, br_abs_confirm=0.015, mean3_abs_confirm=0.006,
               pass_abs_fail=0.010, pressure_abs_fail=0.012, br_abs_fail=0.020, mean3_abs_fail=0.010),
    "C4": dict(words=["holds", "does not hold", "inconclusive"], u1_dest_gate_min=0.60, u1_dest_rank=1, shuf_dest_gate_max=0.35, shuf_dest_rank=6, shuf_fail_rank_at_most=4),
}
MEAN3 = ["pass", "ball_recovery", "pressure"]


def no_dup(keys, what):
    s = pd.Series(list(keys))
    d = s[s.duplicated()]
    if len(d):
        raise RuntimeError(f"{what}  the same key occurs twice  {list(d.unique())[:5]}")


def auc_weighted_sorted(order, y, w):
    """AUC under integer match weights w for scores sorted in advance (unused, AUCFast does this)."""
    raise NotImplementedError


class AUCFast:
    """Sorts the scores once; each resample then computes the AUC from the weights in O(n) (Mann Whitney, ties count 0.5)."""

    def __init__(self, score, y):
        self.n = len(score)
        o = np.argsort(score, kind="mergesort")
        s = score[o]
        self.o = o
        self.y = y[o].astype(bool)
        # groups of tied scores
        grp = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])])
        self.grp = grp
        self.ng = int(grp[-1]) + 1

    def auc(self, w):
        w = w[self.o].astype(np.float64)
        wp = np.where(self.y, w, 0.0)
        wn = np.where(self.y, 0.0, w)
        # positive and negative weight within each tie group
        gp = np.bincount(self.grp, wp, minlength=self.ng)
        gn = np.bincount(self.grp, wn, minlength=self.ng)
        cn_below = np.concatenate([[0.0], np.cumsum(gn)[:-1]])
        W1, W2 = gp.sum(), gn.sum()
        if W1 == 0 or W2 == 0:
            return float("nan")
        return float((gp * (cn_below + 0.5 * gn)).sum() / (W1 * W2))


def load_holdout(mode):
    """Returns a list of runs, each dict(group, model, seed, task, score, y, match, top1, logp, gate).
    group is unified, single, single_rich or rules."""
    runs = []
    if mode == "check_2425":
        clean = clean_matches()
        for tag, seeds in UNIFIED_TAGS.items():
            for sd in seeds:
                f = U1 / f"{tag}_s{sd}.npz"
                if not f.exists():
                    continue
                z = np.load(f, allow_pickle=True)
                ok = np.array([int(m) in clean for m in z["match_id"]])
                for t in TASKS6:
                    m = (z["task"] == TASK_ID[t]) & ok
                    runs.append(dict(group="unified", model=tag, seed=sd, task=t, score=z["logit_bin"][m], y=z["y"][m].astype(int), match=z["match_id"][m].astype(int),
                                     top1=z["dest_top1"][m], logp=z["dest_logp_true"][m], gate=z["gate_mean"][m]))
        for grp, models in SINGLE.items():
            d = Path("data/cache") / ("single_clean" if grp == "single" else "single_clean_rich") / "preds"
            for t in TASKS6:
                mf = d / f"{t}_test_meta.npz"
                if not mf.exists():
                    continue
                meta = np.load(mf, allow_pickle=True)
                ok = np.array([int(m) in clean for m in meta["match_id"]])
                for model, seeds in models.items():
                    for sd in seeds:
                        f = d / f"{t}_{model}_s{sd}.npz"
                        if not f.exists():
                            continue
                        z = np.load(f, allow_pickle=True)
                        runs.append(dict(group=grp, model=model, seed=sd, task=t, score=z["prob"][ok] if "prob" in z.files else None,
                                         y=meta["y"][ok].astype(int), match=meta["match_id"][ok].astype(int),
                                         top1=z["top1"][ok] if "top1" in z.files else None, logp=z["logp_true"][ok] if "logp_true" in z.files else None,
                                         gate=z["gate"][ok] if "gate" in z.files else None))
        return runs
    for tag, seeds in UNIFIED_TAGS.items():
        for sd in seeds:
            f = HOLD / "unified" / f"{tag}_s{sd}_holdout.npz"
            if not f.exists():
                print(f"[miss] {f}", flush=True)
                continue
            z = np.load(f, allow_pickle=True)
            for t in TASKS6:
                m = z["task"] == TASK_ID[t]
                runs.append(dict(group="unified", model=tag, seed=sd, task=t, score=z["logit_bin"][m], y=z["y"][m].astype(int), match=z["match_id"][m].astype(int),
                                 top1=z["dest_top1"][m], logp=z["dest_logp_true"][m], gate=z["gate_mean"][m]))
    for grp, models in SINGLE.items():
        d = HOLD / grp / "holdout_preds"
        for t in TASKS6:
            mf = d / f"{t}_holdout_meta.npz"
            if not mf.exists():
                print(f"[miss] {mf}", flush=True)
                continue
            meta = np.load(mf, allow_pickle=True)
            for model, seeds in models.items():
                for sd in seeds:
                    f = d / f"{t}_{model}_s{sd}.npz"
                    if not f.exists():
                        continue
                    z = np.load(f, allow_pickle=True)
                    runs.append(dict(group=grp, model=model, seed=sd, task=t, score=z["prob"] if "prob" in z.files else None, y=meta["y"].astype(int), match=meta["match_id"].astype(int),
                                     top1=z["top1"] if "top1" in z.files else None, logp=z["logp_true"] if "logp_true" in z.files else None, gate=z["gate"] if "gate" in z.files else None))
    rf = HOLD / "dest_rules_holdout.npz"
    if rf.exists():
        z = np.load(rf, allow_pickle=True)
        names = json.load(open(RULES_JSON))["_rule_order"]
        for i, name in enumerate(names):
            runs.append(dict(group="rules", model=name, seed=0, task="dest", score=None, y=z["y"].astype(int), match=z["match_id"].astype(int),
                             top1=z[f"rule{i}_top1"], logp=z[f"rule{i}_logp_true"], gate=None))
    else:
        print(f"[miss] {rf}", flush=True)
    no_dup(((r["group"], r["model"], r["seed"], r["task"]) for r in runs), "run table")
    return runs


def point_metrics(r, w=None):
    """Metrics of one run on one task. w are the match resampling weights (None means unweighted)."""
    y = r["y"]
    w = np.ones(len(y)) if w is None else w
    out = {}
    if r["task"] == "dest":
        if r["top1"] is not None:
            out["top1"] = float(np.average(r["top1"] == y, weights=w))
            nll = float(-np.average(r["logp"], weights=w))
            out["logloss"] = nll
            out["skill"] = 1 - nll / math.log(96)
    else:
        if r["score"] is not None:
            if "auc_fast" not in r:
                r["auc_fast"] = AUCFast(np.asarray(r["score"], dtype=np.float64), y)
            out["auc"] = r["auc_fast"].auc(w)
    if r["gate"] is not None and len(r["gate"]) and not np.all(np.isnan(r["gate"])):
        out["gate"] = float(np.average(np.nan_to_num(r["gate"]), weights=w))
    return out


def key_metric(task):
    return "top1" if task == "dest" else "auc"


def aggregate(runs, w=None):
    """{(group, model, task): {metric: (mean over seeds, sd, n, {seed: value})}}"""
    per = {}
    for r in runs:
        pm = point_metrics(r, w)
        k = (r["group"], r["model"], r["task"])
        per.setdefault(k, {})
        for mname, v in pm.items():
            per[k].setdefault(mname, {})
            if r["seed"] in per[k][mname]:
                raise RuntimeError(f"the same key occurs twice  {k} {mname} seed {r['seed']}")
            per[k][mname][r["seed"]] = v
    agg = {}
    for k, ms in per.items():
        agg[k] = {}
        for mname, sv in ms.items():
            vals = np.array(list(sv.values()), dtype=float)
            agg[k][mname] = dict(mean=float(np.nanmean(vals)), sd=float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan"), n=int(len(vals)), by_seed=sv)
    return agg


def paired_diff(agg, ka, kb, metric, paired=True):
    """a minus b. When paired, the mean of same-seed differences (single-task models, where a seed is the same data split);
    otherwise the difference of seed means (unified model against single-task models, as specified for C3)."""
    a, b = agg.get(ka, {}).get(metric), agg.get(kb, {}).get(metric)
    if a is None or b is None:
        return None, {}, "missing"
    common = sorted(set(a["by_seed"]) & set(b["by_seed"])) if paired else []
    if common:
        d = {s: a["by_seed"][s] - b["by_seed"][s] for s in common}
        return float(np.mean(list(d.values()))), d, f"paired over {len(common)} seeds"
    return a["mean"] - b["mean"], {}, f"difference of means ({a['n']} against {b['n']} seeds, not paired)"


def derived(agg):
    """All derived quantities used by the verdicts, computed here and recomputed in the same way for every bootstrap resample."""
    D = {}
    # C1
    fcn = agg.get(("unified", "u1fcn", "dest"), {}).get("top1")
    rules = {m: v["top1"]["mean"] for (g, m, t), v in agg.items() if g == "rules" and t == "dest" and "top1" in v}
    best_rule = max(rules.items(), key=lambda kv: kv[1]) if rules else (None, float("nan"))
    m2r = agg.get(("single_rich", "M2_MLP_360", "dest"), {}).get("top1")
    m2t = agg.get(("single", "M2_MLP_360", "dest"), {}).get("top1")
    D["C1"] = dict(fcn_mean=fcn["mean"] if fcn else float("nan"), fcn_by_seed=fcn["by_seed"] if fcn else {}, best_rule=best_rule[0], best_rule_top1=best_rule[1],
                   m2rich_mean=m2r["mean"] if m2r else float("nan"), m2thin_mean=m2t["mean"] if m2t else float("nan"))
    D["C1"]["fcn_minus_rule"] = D["C1"]["fcn_mean"] - best_rule[1]
    D["C1"]["fcn_minus_m2rich"] = D["C1"]["fcn_mean"] - D["C1"]["m2rich_mean"]
    # C2
    c2 = {k: {} for k in C2_KEYS}
    for t in OUTCOME5:
        c2["thin_gain"][t] = paired_diff(agg, ("single", "M4_CNN_Full", t), ("single", "M2_MLP_360", t), "auc")[0]
        c2["rich_gain"][t] = paired_diff(agg, ("single_rich", "M4_CNN_Full", t), ("single_rich", "M2_MLP_360", t), "auc")[0]
        c2["xgbrich_minus_m4thin"][t] = paired_diff(agg, ("single_rich", "B2_XGB_360", t), ("single", "M4_CNN_Full", t), "auc")[0]
        c2["m4rich_minus_xgbrich"][t] = paired_diff(agg, ("single_rich", "M4_CNN_Full", t), ("single_rich", "B2_XGB_360", t), "auc")[0]
        c2["m4thin_minus_xgbthin"][t] = paired_diff(agg, ("single", "M4_CNN_Full", t), ("single", "B2_XGB_360", t), "auc")[0]
        c2["xgbrich_minus_xgbthin"][t] = paired_diff(agg, ("single_rich", "B2_XGB_360", t), ("single", "B2_XGB_360", t), "auc")[0]
    for k in C2_KEYS:
        vals = [v for v in c2[k].values() if v is not None]
        c2[k + "_mean5"] = float(np.mean(vals)) if len(vals) == 5 else float("nan")
        vals3 = [c2[k][t] for t in MEAN3 if c2[k].get(t) is not None]
        c2[k + "_mean3"] = float(np.mean(vals3)) if len(vals3) == 3 else float("nan")
    c2["rich_over_thin_ratio3"] = c2["rich_gain_mean3"] / c2["thin_gain_mean3"] if c2["thin_gain_mean3"] else float("nan")
    c2["rich_over_thin_ratio5"] = c2["rich_gain_mean5"] / c2["thin_gain_mean5"] if c2["thin_gain_mean5"] else float("nan")
    D["C2"] = c2
    # C3
    c3 = dict(u1_minus_xgbrich={}, u1rich_minus_xgbrich={}, u1fcn_minus_u1={}, u1rich_minus_u1={}, u1concat_minus_u1={})
    for t in OUTCOME5:
        c3["u1_minus_xgbrich"][t] = paired_diff(agg, ("unified", "u1", t), ("single_rich", "B2_XGB_360", t), "auc", paired=False)[0]
        c3["u1rich_minus_xgbrich"][t] = paired_diff(agg, ("unified", "u1rich", t), ("single_rich", "B2_XGB_360", t), "auc", paired=False)[0]
        c3["u1fcn_minus_u1"][t] = paired_diff(agg, ("unified", "u1fcn", t), ("unified", "u1", t), "auc", paired=False)[0]
        c3["u1rich_minus_u1"][t] = paired_diff(agg, ("unified", "u1rich", t), ("unified", "u1", t), "auc", paired=False)[0]
        c3["u1concat_minus_u1"][t] = paired_diff(agg, ("unified", "u1concat", t), ("unified", "u1", t), "auc", paired=False)[0]
    vals = [v for v in c3["u1_minus_xgbrich"].values() if v is not None]
    c3["u1_minus_xgbrich_mean5"] = float(np.mean(vals)) if len(vals) == 5 else float("nan")
    vals3 = [c3["u1_minus_xgbrich"][t] for t in ("pass", "ball_recovery", "pressure") if c3["u1_minus_xgbrich"].get(t) is not None]
    c3["u1_minus_xgbrich_mean3"] = float(np.mean(vals3)) if len(vals3) == 3 else float("nan")
    D["C3"] = c3
    # C4, gate rank among the six tasks
    c4 = {}
    for tag in ("u1", "shuf_dest", "u1rich"):
        per_seed = {}
        seeds = set()
        for (g, m, t), v in agg.items():
            if g == "unified" and m == tag and "gate" in v:
                seeds |= set(v["gate"]["by_seed"])
        for sd in sorted(seeds):
            gates = {t: agg[("unified", tag, t)]["gate"]["by_seed"][sd] for t in TASKS6 if ("unified", tag, t) in agg and sd in agg[("unified", tag, t)]["gate"]["by_seed"]}
            if len(gates) < 6:
                continue
            order = sorted(gates, key=lambda t: -gates[t])
            per_seed[sd] = dict(gates=gates, dest_rank=order.index("dest") + 1, dest_gate=gates["dest"])
        c4[tag] = per_seed
    D["C4"] = c4
    return D


def judge(D):
    R = RULES
    J = {}
    c1 = D["C1"]
    seeds_above_rule = all(v > c1["best_rule_top1"] for v in c1["fcn_by_seed"].values()) if c1["fcn_by_seed"] else False
    if np.isnan(c1["fcn_minus_rule"]) or np.isnan(c1["fcn_minus_m2rich"]):
        J["C1"] = "missing data"
    elif c1["fcn_minus_rule"] >= R["C1"]["fcn_minus_rule_confirm"] and c1["fcn_minus_m2rich"] >= R["C1"]["fcn_minus_m2rich_confirm"] and seeds_above_rule:
        J["C1"] = "confirmed"
    elif c1["fcn_minus_rule"] <= R["C1"]["fcn_minus_rule_fail"] or c1["fcn_minus_m2rich"] <= R["C1"]["fcn_minus_m2rich_fail"]:
        J["C1"] = "not confirmed"
    else:
        J["C1"] = "inconclusive"
    c2 = D["C2"]
    pt, pr = c2["thin_gain"].get("pressure"), c2["rich_gain"].get("pressure")
    if any(np.isnan(c2[k]) for k in ("thin_gain_mean3", "rich_gain_mean3", "xgbrich_minus_m4thin_mean3")) or pt is None or pr is None:
        J["C2"] = "missing data"
    elif (c2["xgbrich_minus_m4thin_mean3"] >= R["C2"]["xgbrich_minus_m4thin_confirm"] and c2["rich_gain_mean3"] <= R["C2"]["rich_gain_ratio_confirm"] * c2["thin_gain_mean3"]
          and c2["rich_gain_mean3"] <= R["C2"]["rich_gain_abs_confirm"] and pt >= R["C2"]["pressure_thin_gain_min"] and pr <= R["C2"]["pressure_rich_gain_max"]):
        J["C2"] = "confirmed"
    elif (c2["xgbrich_minus_m4thin_mean3"] <= R["C2"]["xgbrich_minus_m4thin_fail"] or c2["rich_gain_mean3"] >= c2["thin_gain_mean3"]
          or pr >= R["C2"]["pressure_rich_over_thin_fail"] * pt):
        J["C2"] = "not confirmed"
    else:
        J["C2"] = "inconclusive"
    c3 = D["C3"]["u1_minus_xgbrich"]
    m3 = D["C3"]["u1_minus_xgbrich_mean3"]
    if any(c3.get(t) is None for t in MEAN3) or np.isnan(m3):
        J["C3"] = "missing data"
    elif (abs(c3["pass"]) <= R["C3"]["pass_abs_confirm"] and abs(c3["pressure"]) <= R["C3"]["pressure_abs_confirm"]
          and abs(c3["ball_recovery"]) <= R["C3"]["br_abs_confirm"] and abs(m3) <= R["C3"]["mean3_abs_confirm"]):
        J["C3"] = "confirmed"
    elif (abs(c3["pass"]) > R["C3"]["pass_abs_fail"] or abs(c3["pressure"]) > R["C3"]["pressure_abs_fail"]
          or abs(c3["ball_recovery"]) > R["C3"]["br_abs_fail"] or abs(m3) > R["C3"]["mean3_abs_fail"]):
        J["C3"] = "not confirmed"
    else:
        J["C3"] = "inconclusive"
    c4 = D["C4"]
    u1, sh = c4.get("u1", {}), c4.get("shuf_dest", {})
    if len(u1) < 4 or len(sh) < 3:
        J["C4"] = f"missing data (u1 {len(u1)} seeds, shuf_dest {len(sh)} seeds)"
    elif (all(v["dest_rank"] == R["C4"]["u1_dest_rank"] and v["dest_gate"] >= R["C4"]["u1_dest_gate_min"] for v in u1.values())
          and all(v["dest_rank"] == R["C4"]["shuf_dest_rank"] and v["dest_gate"] <= R["C4"]["shuf_dest_gate_max"] for v in sh.values())):
        J["C4"] = "holds"
    elif any(v["dest_rank"] <= R["C4"]["shuf_fail_rank_at_most"] for v in sh.values()) or any(v["dest_rank"] != 1 for v in u1.values()):
        J["C4"] = "does not hold"
    else:
        J["C4"] = "inconclusive"
    return J


def bootstrap(runs, n_boot):
    """Resamples matches; returns the 2.5 and 97.5 percentiles of every derived quantity and of the main-metric mean of every (group, model, task)."""
    matches = np.unique(np.concatenate([r["match"] for r in runs]))
    rng = np.random.RandomState(BOOT_SEED)
    idx = {m: i for i, m in enumerate(matches)}
    for r in runs:
        r["_mi"] = np.array([idx[m] for m in r["match"]])
    coll_D, coll_A = [], []
    t0 = time.time()
    for b in range(n_boot):
        counts = np.bincount(rng.randint(0, len(matches), len(matches)), minlength=len(matches)).astype(np.float64)
        agg_b = aggregate(runs, None)  # placeholder, replaced below by the weighted metrics of every run
        agg_b = {}
        per = {}
        for r in runs:
            w = counts[r["_mi"]]
            pm = point_metrics(r, w)
            k = (r["group"], r["model"], r["task"])
            per.setdefault(k, {})
            for mname, v in pm.items():
                per[k].setdefault(mname, {})[r["seed"]] = v
        for k, ms in per.items():
            agg_b[k] = {mname: dict(mean=float(np.nanmean(list(sv.values()))), sd=float("nan"), n=len(sv), by_seed=sv) for mname, sv in ms.items()}
        coll_D.append(derived(agg_b))
        coll_A.append({k: v[key_metric(k[2])]["mean"] for k, v in agg_b.items() if key_metric(k[2]) in v})
        if (b + 1) % 100 == 0:
            el = time.time() - t0
            print(f"[boot] {b + 1}/{n_boot}  {el / 60:.1f} min  eta {el / (b + 1) * (n_boot - b - 1) / 60:.0f} min", flush=True)
    def q(vals):
        v = np.array(vals, dtype=float)
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    ci = dict(metrics={}, derived={})
    for k in coll_A[0]:
        ci["metrics"]["|".join(k)] = q([a.get(k, np.nan) for a in coll_A])
    ci["derived"]["C1"] = {kk: q([d["C1"][kk] for d in coll_D]) for kk in ("fcn_mean", "best_rule_top1", "m2rich_mean", "m2thin_mean", "fcn_minus_rule", "fcn_minus_m2rich")}
    ci["derived"]["C2"] = {}
    for kk in C2_KEYS:
        ci["derived"]["C2"][kk] = {t: q([d["C2"][kk].get(t, np.nan) if d["C2"][kk].get(t) is not None else np.nan for d in coll_D]) for t in OUTCOME5}
        ci["derived"]["C2"][kk + "_mean5"] = q([d["C2"][kk + "_mean5"] for d in coll_D])
        ci["derived"]["C2"][kk + "_mean3"] = q([d["C2"][kk + "_mean3"] for d in coll_D])
    ci["derived"]["C3"] = {}
    for kk in ("u1_minus_xgbrich", "u1rich_minus_xgbrich", "u1fcn_minus_u1", "u1rich_minus_u1", "u1concat_minus_u1"):
        ci["derived"]["C3"][kk] = {t: q([d["C3"][kk].get(t, np.nan) if d["C3"][kk].get(t) is not None else np.nan for d in coll_D]) for t in OUTCOME5}
    ci["derived"]["C3"]["u1_minus_xgbrich_mean5"] = q([d["C3"]["u1_minus_xgbrich_mean5"] for d in coll_D])
    ci["derived"]["C3"]["u1_minus_xgbrich_mean3"] = q([d["C3"]["u1_minus_xgbrich_mean3"] for d in coll_D])
    return ci


def fmt(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fmt_ci(ci, nd=4):
    return f"[{fmt(ci[0], nd)}, {fmt(ci[1], nd)}]" if ci else ""


def ref_2425():
    """Reference values of 2024/25, fixed together with the thresholds before the held-out run; shown side by side only."""
    return dict(C1=dict(fcn_mean=0.2177, best_rule_top1=0.1578, m2rich_mean=0.1004, m2thin_mean=0.0947),
                C2=dict(thin_gain={"pass": 0.0031, "shot": 0.0030, "interception": 0.0158, "ball_recovery": 0.0058, "pressure": 0.0255},
                        rich_gain={"pass": 0.0016, "shot": 0.0031, "interception": 0.0112, "ball_recovery": 0.0046, "pressure": 0.0024},
                        xgbrich_minus_m4thin={"pass": 0.0083, "shot": -0.0014, "interception": 0.0043, "ball_recovery": 0.0026, "pressure": -0.0034},
                        m4rich_minus_xgbrich={"pass": -0.0070, "shot": 0.0026, "interception": 0.0093, "ball_recovery": 0.0015, "pressure": 0.0015},
                        m4thin_minus_xgbthin={"pass": -0.0050, "shot": 0.0011, "interception": 0.0159, "ball_recovery": 0.0030, "pressure": 0.0260},
                        xgbrich_minus_xgbthin={"pass": 0.0033, "shot": -0.0002, "interception": 0.0202, "ball_recovery": 0.0057, "pressure": 0.0225}),
                C3=dict(u1_minus_xgbrich={"pass": -0.0039, "shot": -0.0022, "interception": 0.0036, "ball_recovery": -0.0004, "pressure": 0.0052},
                        u1rich_minus_xgbrich={"pass": -0.0042, "shot": -0.0039, "interception": 0.0006, "ball_recovery": -0.0008, "pressure": 0.0036}))


def write_md(runs, agg, D, J, ci, mode, n_boot):
    L = [f"# Held-out confirmation on 25/26, {'summary of the 146 matches' if mode != 'check_2425' else 'check mode on the 202 matches of 24/25'}", "",
         "Generated by `scripts/eval/holdout_2526_summarize.py`. The thresholds below were fixed before the held-out run (see `results/decision_rules.md`).", "",
         "```", json.dumps(RULES, ensure_ascii=False, indent=1), "```", ""]
    if mode == "check_2425":
        L += ["Check mode. The column headed Held-out holds the values this script recomputes from the original 2024/25 predictions on the 202 test matches, "
              "to be compared with the 24/25 column and with `results/u1_clean_summary.md`. No verdicts are made, and the rule baselines, bootstrap intervals "
              "and league split are not computed in this mode.", ""]
    else:
        L += ["## Verdicts", "", "| Claim | Verdict |", "|---|---|"] + [f"| {k} | **{v}** |" for k, v in J.items()] + [""]
    ref = ref_2425()
    c1 = D["C1"]
    L += ["## C1. The spatial structure cannot be replaced for pass destination", "", "| Quantity | Held-out | 95% CI | 24/25 |", "|---|---:|---|---:|",
          f"| u1fcn mean of three seeds, top-1 | {fmt(c1['fcn_mean'])} | {fmt_ci(ci['derived']['C1']['fcn_mean']) if ci else ''} | {ref['C1']['fcn_mean']} |",
          f"| u1fcn per seed | {' '.join(fmt(v) for v in c1['fcn_by_seed'].values())} | | 0.215 0.217 0.221 |",
          f"| Strongest rule ({c1['best_rule']}) | {fmt(c1['best_rule_top1'])} | {fmt_ci(ci['derived']['C1']['best_rule_top1']) if ci else ''} | {ref['C1']['best_rule_top1']} |",
          f"| M2 enriched | {fmt(c1['m2rich_mean'])} | {fmt_ci(ci['derived']['C1']['m2rich_mean']) if ci else ''} | {ref['C1']['m2rich_mean']} |",
          f"| M2 thin | {fmt(c1['m2thin_mean'])} | {fmt_ci(ci['derived']['C1']['m2thin_mean']) if ci else ''} | {ref['C1']['m2thin_mean']} |",
          f"| u1fcn minus strongest rule | {fmt(c1['fcn_minus_rule'])} | {fmt_ci(ci['derived']['C1']['fcn_minus_rule']) if ci else ''} | 0.0599 |",
          f"| u1fcn minus M2 enriched | {fmt(c1['fcn_minus_m2rich'])} | {fmt_ci(ci['derived']['C1']['fcn_minus_m2rich']) if ci else ''} | 0.1173 |", ""]
    if c1["best_rule"] is None:  # check mode, no rule baselines
        L = [x for x in L if not x.startswith(("| Strongest rule (", "| u1fcn minus strongest rule |"))]
    c2 = D["C2"]
    L += ["## C2. Enriched scalars replace most of the spatial gain (judged on the mean of three tasks, Pass, Ball Recovery, Pressure)", "", "| Quantity | " + " | ".join(OUTCOME5) + " | Three-task mean | Five-task mean |", "|---|" + "---:|" * 7]
    for kk, label in (("thin_gain", "Thin gain M4 minus M2"), ("rich_gain", "Enriched gain M4 minus M2"), ("xgbrich_minus_m4thin", "XGB enriched minus M4 thin"), ("m4rich_minus_xgbrich", "M4 enriched minus XGB enriched"),
                      ("m4thin_minus_xgbthin", "M4 thin minus XGB thin (reported only)"), ("xgbrich_minus_xgbthin", "XGB enriched minus XGB thin (reported only)")):
        L.append(f"| {label}, held-out | " + " | ".join(fmt(c2[kk].get(t)) for t in OUTCOME5) + f" | {fmt(c2[kk + '_mean3'])} | {fmt(c2[kk + '_mean5'])} |")
        if ci:
            L.append(f"| Same, 95% CI | " + " | ".join(fmt_ci(ci['derived']['C2'][kk][t]) for t in OUTCOME5) + f" | {fmt_ci(ci['derived']['C2'][kk + '_mean3'])} | {fmt_ci(ci['derived']['C2'][kk + '_mean5'])} |")
        L.append(f"| Same, 24/25 | " + " | ".join(fmt(ref['C2'][kk][t]) for t in OUTCOME5) + f" | {fmt(np.mean([ref['C2'][kk][t] for t in MEAN3]))} | {fmt(np.mean(list(ref['C2'][kk].values())))} |")
    L += [f"| Enriched gain mean over thin gain mean | | | | | | {fmt(c2['rich_over_thin_ratio3'], 2)} (0.25 in 24/25) | {fmt(c2['rich_over_thin_ratio5'], 2)} (0.43 in 24/25) |", ""]
    c3 = D["C3"]
    L += ["## C3. The unified model matches enriched XGBoost (judged on Pass, Pressure, Ball Recovery and the three-task mean)", "", "| Quantity | " + " | ".join(OUTCOME5) + " | Five-task mean | Three-task mean (Pass BR Pressure) |", "|---|" + "---:|" * 7]
    for kk, label in (("u1_minus_xgbrich", "u1 minus XGB enriched"), ("u1rich_minus_xgbrich", "u1rich minus XGB enriched"), ("u1fcn_minus_u1", "u1fcn minus u1"), ("u1rich_minus_u1", "u1rich minus u1"), ("u1concat_minus_u1", "u1concat minus u1")):
        m5 = fmt(c3["u1_minus_xgbrich_mean5"]) if kk == "u1_minus_xgbrich" else ""
        m3 = fmt(c3["u1_minus_xgbrich_mean3"]) if kk == "u1_minus_xgbrich" else ""
        L.append(f"| {label}, held-out | " + " | ".join(fmt(c3[kk].get(t)) for t in OUTCOME5) + f" | {m5} | {m3} |")
        if ci:
            L.append(f"| Same, 95% CI | " + " | ".join(fmt_ci(ci['derived']['C3'][kk][t]) for t in OUTCOME5)
                     + (f" | {fmt_ci(ci['derived']['C3']['u1_minus_xgbrich_mean5'])} | {fmt_ci(ci['derived']['C3']['u1_minus_xgbrich_mean3'])} |" if kk == "u1_minus_xgbrich" else " | | |"))
        if kk in ref["C3"]:
            L.append(f"| Same, 24/25 | " + " | ".join(fmt(ref['C3'][kk][t]) for t in OUTCOME5) + f" | {fmt(np.mean(list(ref['C3'][kk].values())))} | {fmt(np.mean([ref['C3'][kk][t] for t in MEAN3]))} |")
    L.append("")
    L += ["## C4. Pattern of the destination gate readings (not a held-out test of the mechanism)", "", "| Run | Seed | Destination gate | Destination rank (among six tasks) | Gate per task |", "|---|---:|---:|---:|---|"]
    for tag, per in D["C4"].items():
        for sd, v in per.items():
            L.append(f"| {tag} | {sd} | {fmt(v['dest_gate'], 3)} | {v['dest_rank']} | " + " ".join(f"{t} {fmt(v['gates'][t], 3)}" for t in TASKS6) + " |")
    L += ["", "24/25 reference, u1 destination gate 0.78 to 0.83 and rank 1 in all four seeds; shuf_dest 0.24 to 0.28 and last in all three seeds.", ""]
    # metrics of every model
    ref_m = json.load(open(REF_JSON)) if (mode != "check_2425" and REF_JSON.exists()) else {}
    L += ["## All models per task (seed mean ± SD, n seeds; held-out set; brackets give the same quantity in 24/25 and held-out minus 24/25)", "", "| Group | Model | " + " | ".join(TASKS6) + " |", "|---|---|" + "---:|" * 6]
    keys = sorted({(g, m) for (g, m, t) in agg}, key=lambda x: (x[0], x[1]))
    for g, m in keys:
        cells = []
        for t in TASKS6:
            v = agg.get((g, m, t), {})
            mm = key_metric(t)
            if mm in v:
                s = f"{fmt(v[mm]['mean'])} ± {fmt(v[mm]['sd'])} n={v[mm]['n']}"
                if t == "dest" and "skill" in v:
                    s += f" (skill {fmt(v['skill']['mean'])})"
                if ci and "|".join((g, m, t)) in ci["metrics"]:
                    s += f" {fmt_ci(ci['metrics']['|'.join((g, m, t))])}"
                rk = "|".join((g, m, t))
                if rk in ref_m and mm in ref_m[rk]:
                    s += f" [24/25 {fmt(ref_m[rk][mm])}, change {fmt(v[mm]['mean'] - ref_m[rk][mm])}]"
                cells.append(s)
            else:
                cells.append("")
        L.append(f"| {g} | {m} | " + " | ".join(cells) + " |")
    L += ["", "The dest column is top-1 (skill in parentheses), the others are AUC. Intervals are 95% bootstrap intervals over matches (" + (f"{n_boot} resamples, RandomState {BOOT_SEED}" if ci else "not computed in this run") + "). Changes in absolute level are expected and do not enter the verdicts.", ""]
    # by league
    n_head = len(L)
    L += ["## By league (main metric, seed mean)", "", "| Group | Model | Task | La Liga | Premier League | La Liga events | Premier League events |", "|---|---|---|---:|---:|---:|---:|"]
    byleague = {}
    for r in runs:
        lg = r.get("league")
        if lg is None:
            continue
        for L_name in ("La Liga", "Premier League"):
            m = lg == L_name
            if m.sum() == 0:
                continue
            sub = dict(r)
            sub.update(y=r["y"][m], score=None if r["score"] is None else r["score"][m], top1=None if r["top1"] is None else r["top1"][m],
                       logp=None if r["logp"] is None else r["logp"][m], gate=None if r["gate"] is None else r["gate"][m])
            sub.pop("auc_fast", None)
            pm = point_metrics(sub)
            k = (r["group"], r["model"], r["task"])
            byleague.setdefault(k, {}).setdefault(L_name, []).append(pm.get(key_metric(r["task"]), np.nan))
            byleague[k].setdefault(L_name + "_n", int(m.sum()))
    for k in sorted(byleague):
        v = byleague[k]
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {fmt(np.nanmean(v.get('La Liga', [np.nan])))} | {fmt(np.nanmean(v.get('Premier League', [np.nan])))} | {v.get('La Liga_n', '')} | {v.get('Premier League_n', '')} |")
    if byleague:
        L.append("")
    else:  # no league labels (check mode)
        del L[n_head:]
    # event counts and label means
    L += [f"## {'Held-out' if mode != 'check_2425' else 'Test'} events and label means (event table of the first seed of unified u1)", "", "| Task | Events | Label mean |", "|---|---:|---:|"]
    seen = set()
    for r in runs:
        if r["group"] == "unified" and r["model"] == "u1" and r["task"] not in seen:
            seen.add(r["task"])
            L.append(f"| {r['task']} | {len(r['y']):,} | {fmt(float(np.mean(r['y'])), 3)} |")
    L.append("")
    out_md = OUT_MD if mode != "check_2425" else Path("results/holdout_2526_summary_check2425.md")
    out_md.write_text("\n".join(L), encoding="utf8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check_2425", action="store_true")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    mode = "check_2425" if a.check_2425 else "holdout"
    t0 = time.time()
    print("[rules] " + json.dumps(RULES, ensure_ascii=False), flush=True)
    runs = load_holdout(mode)
    print(f"[load] {len(runs)} run x task entries  ({time.time() - t0:.0f}s)", flush=True)
    if mode != "check_2425":
        lg = pd.read_parquet(HOLD / "match_table_2526.parquet", columns=["match_id", "season_dir"])
        lgmap = {int(m): ("La Liga" if s.startswith("11_") else "Premier League") for m, s in zip(lg["match_id"], lg["season_dir"])}
        for r in runs:
            r["league"] = np.array([lgmap[int(m)] for m in r["match"]])
    agg = aggregate(runs)
    D = derived(agg)
    J = judge(D) if mode != "check_2425" else {}
    ci = bootstrap(runs, a.n_boot) if (mode != "check_2425" and a.n_boot > 0) else None
    if mode != "check_2425":
        out = dict(rules=RULES, judgement=J, derived=json.loads(json.dumps(D, default=lambda o: None)),
                   metrics={"|".join(k): {m: dict(mean=v["mean"], sd=v["sd"], n=v["n"], by_seed={str(s): x for s, x in v["by_seed"].items()}) for m, v in ms.items()} for k, ms in agg.items()},
                   ci=ci, n_boot=a.n_boot)
        json.dump(out, open(SUMMARY_JSON, "w"), ensure_ascii=False, indent=1)
    if mode == "check_2425":
        json.dump({"|".join(k): {m: v["mean"] for m, v in ms.items()} for k, ms in agg.items()}, open(REF_JSON, "w"), ensure_ascii=False, indent=1)
        print(f"[ref] 24/25 means of every model saved to {REF_JSON}", flush=True)
    write_md(runs, agg, D, J, ci, mode, a.n_boot)
    for k, v in J.items():
        print(f"[judge] {k}  {v}", flush=True)
    for k in sorted(agg):
        mm = key_metric(k[2])
        if mm in agg[k]:
            print(f"[agg] {k[0]:12s} {k[1]:36s} {k[2]:14s} {mm} {agg[k][mm]['mean']:.4f} ± {fmt(agg[k][mm]['sd'])} n={agg[k][mm]['n']}")
    print(f"[done] {OUT_MD if mode != 'check_2425' else 'results/holdout_2526_summary_check2425.md'}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
