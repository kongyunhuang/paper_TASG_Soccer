#!/usr/bin/env python3
"""
留出数据确认，汇总和判定  读留出集上的逐事件预测，算指标、自助法区间，按冻结方案的阈值机械判定
============================
阈值字典 RULES 和 memory_logs/2026-10-02_留出数据确认_冻结方案和判定标准.md 第二节逐字对应，运行时原样打印。
分组和配对的键（run、seed、task、model）出现两次就报错，不静默覆盖。
  --check_2425  不读留出集，改读 24/25 的原预测（u1_clean 的 npz、single_clean{_rich}/preds），限制到 202 场干净比赛，
                算同一套指标，用来核对本脚本的指标算法和 audit/u1_clean_summary.md 一致。不做判定、不做自助法。
  默认          读 data/cache/holdout_2526/ 下第二阶段的输出，算指标、按比赛自助法（146 场重抽，--n_boot 次，RandomState 20261003）、判定，
                写 audit/holdout_2526_summary.md 和 data/cache/holdout_2526/summary.json。

输入  data/cache/holdout_2526/unified/{tag}_s{seed}_holdout.npz
      data/cache/holdout_2526/single{_rich}/holdout_preds/{task}_{model}_s{seed}.npz 和 {task}_holdout_meta.npz
      data/cache/holdout_2526/dest_rules_holdout.npz 和 .json
输出  audit/holdout_2526_summary.md，data/cache/holdout_2526/summary.json

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/eval/holdout_2526_summarize.py --check_2425
  PYTHONPATH=. python -u scripts/eval/holdout_2526_summarize.py --n_boot 2000

Last modified 2026-10-02
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
OUT_MD = Path("audit/holdout_2526_summary.md")
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

# 冻结方案第二节的阈值，一字不改
RULES = {
    "C1": dict(fcn_minus_rule_confirm=0.030, fcn_minus_m2rich_confirm=0.060, fcn_minus_rule_fail=0.010, fcn_minus_m2rich_fail=0.020),
    "C2": dict(mean_tasks=["pass", "ball_recovery", "pressure"], xgbrich_minus_m4thin_confirm=-0.003, rich_gain_ratio_confirm=0.5, rich_gain_abs_confirm=0.010,
               pressure_thin_gain_min=0.010, pressure_rich_gain_max=0.010,
               xgbrich_minus_m4thin_fail=-0.008, pressure_rich_over_thin_fail=0.5),
    "C3": dict(mean_tasks=["pass", "ball_recovery", "pressure"], pass_abs_confirm=0.006, pressure_abs_confirm=0.006, br_abs_confirm=0.015, mean3_abs_confirm=0.006,
               pass_abs_fail=0.010, pressure_abs_fail=0.012, br_abs_fail=0.020, mean3_abs_fail=0.010),
    "C4": dict(words=["保持", "不保持", "不确定"], u1_dest_gate_min=0.60, u1_dest_rank=1, shuf_dest_gate_max=0.35, shuf_dest_rank=6, shuf_fail_rank_at_most=4),
}
MEAN3 = ["pass", "ball_recovery", "pressure"]


def no_dup(keys, what):
    s = pd.Series(list(keys))
    d = s[s.duplicated()]
    if len(d):
        raise RuntimeError(f"{what}  同一个键出现两次  {list(d.unique())[:5]}")


def auc_weighted_sorted(order, y, w):
    """按比赛重抽样权重 w（整数次数）的 AUC。order 是分数升序的索引（预先排好，含并列分组信息见 auc_prepare）。"""
    raise NotImplementedError


class AUCFast:
    """分数只排一次，每次重抽样用权重 O(n) 算 AUC（Mann Whitney，带并列 0.5）。"""

    def __init__(self, score, y):
        self.n = len(score)
        o = np.argsort(score, kind="mergesort")
        s = score[o]
        self.o = o
        self.y = y[o].astype(bool)
        # 并列分组
        grp = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])])
        self.grp = grp
        self.ng = int(grp[-1]) + 1

    def auc(self, w):
        w = w[self.o].astype(np.float64)
        wp = np.where(self.y, w, 0.0)
        wn = np.where(self.y, 0.0, w)
        # 每组内正负权重合计
        gp = np.bincount(self.grp, wp, minlength=self.ng)
        gn = np.bincount(self.grp, wn, minlength=self.ng)
        cn_below = np.concatenate([[0.0], np.cumsum(gn)[:-1]])
        W1, W2 = gp.sum(), gn.sum()
        if W1 == 0 or W2 == 0:
            return float("nan")
        return float((gp * (cn_below + 0.5 * gn)).sum() / (W1 * W2))


def load_holdout(mode):
    """返回 runs 列表，每项 dict(group, tag/model, seed, task, score, y, match, top1, logp, gate)。
    group 是 unified、single、single_rich、rules。"""
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
        names = json.load(open(HOLD / "dest_rules_holdout.json"))["_rule_order"]
        for i, name in enumerate(names):
            runs.append(dict(group="rules", model=name, seed=0, task="dest", score=None, y=z["y"].astype(int), match=z["match_id"].astype(int),
                             top1=z[f"rule{i}_top1"], logp=z[f"rule{i}_logp_true"], gate=None))
    else:
        print(f"[miss] {rf}", flush=True)
    no_dup(((r["group"], r["model"], r["seed"], r["task"]) for r in runs), "运行表")
    return runs


def point_metrics(r, w=None):
    """一个运行在一个任务上的指标。w 是按比赛重抽样的权重（None 表示不加权）。"""
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
                raise RuntimeError(f"同一个键出现两次  {k} {mname} seed {r['seed']}")
            per[k][mname][r["seed"]] = v
    agg = {}
    for k, ms in per.items():
        agg[k] = {}
        for mname, sv in ms.items():
            vals = np.array(list(sv.values()), dtype=float)
            agg[k][mname] = dict(mean=float(np.nanmean(vals)), sd=float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan"), n=int(len(vals)), by_seed=sv)
    return agg


def paired_diff(agg, ka, kb, metric, paired=True):
    """a 减 b。paired 时按同种子配对取均值（单任务模型之间，同种子是同一个划分）；不配对时是种子均值之差（统一模型对单任务模型，方案第二节 C3 的写法）。"""
    a, b = agg.get(ka, {}).get(metric), agg.get(kb, {}).get(metric)
    if a is None or b is None:
        return None, {}, "缺"
    common = sorted(set(a["by_seed"]) & set(b["by_seed"])) if paired else []
    if common:
        d = {s: a["by_seed"][s] - b["by_seed"][s] for s in common}
        return float(np.mean(list(d.values()))), d, f"配对 {len(common)} 种子"
    return a["mean"] - b["mean"], {}, f"均值差（{a['n']} 对 {b['n']} 种子，不配对）"


def derived(agg):
    """判定要用的派生量，全部在这里算，自助法时对每个重抽样同样调用。"""
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
    # C4  门控名次（六个任务里）
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
        J["C1"] = "缺数据"
    elif c1["fcn_minus_rule"] >= R["C1"]["fcn_minus_rule_confirm"] and c1["fcn_minus_m2rich"] >= R["C1"]["fcn_minus_m2rich_confirm"] and seeds_above_rule:
        J["C1"] = "确认"
    elif c1["fcn_minus_rule"] <= R["C1"]["fcn_minus_rule_fail"] or c1["fcn_minus_m2rich"] <= R["C1"]["fcn_minus_m2rich_fail"]:
        J["C1"] = "未确认"
    else:
        J["C1"] = "不确定"
    c2 = D["C2"]
    pt, pr = c2["thin_gain"].get("pressure"), c2["rich_gain"].get("pressure")
    if any(np.isnan(c2[k]) for k in ("thin_gain_mean3", "rich_gain_mean3", "xgbrich_minus_m4thin_mean3")) or pt is None or pr is None:
        J["C2"] = "缺数据"
    elif (c2["xgbrich_minus_m4thin_mean3"] >= R["C2"]["xgbrich_minus_m4thin_confirm"] and c2["rich_gain_mean3"] <= R["C2"]["rich_gain_ratio_confirm"] * c2["thin_gain_mean3"]
          and c2["rich_gain_mean3"] <= R["C2"]["rich_gain_abs_confirm"] and pt >= R["C2"]["pressure_thin_gain_min"] and pr <= R["C2"]["pressure_rich_gain_max"]):
        J["C2"] = "确认"
    elif (c2["xgbrich_minus_m4thin_mean3"] <= R["C2"]["xgbrich_minus_m4thin_fail"] or c2["rich_gain_mean3"] >= c2["thin_gain_mean3"]
          or pr >= R["C2"]["pressure_rich_over_thin_fail"] * pt):
        J["C2"] = "未确认"
    else:
        J["C2"] = "不确定"
    c3 = D["C3"]["u1_minus_xgbrich"]
    m3 = D["C3"]["u1_minus_xgbrich_mean3"]
    if any(c3.get(t) is None for t in MEAN3) or np.isnan(m3):
        J["C3"] = "缺数据"
    elif (abs(c3["pass"]) <= R["C3"]["pass_abs_confirm"] and abs(c3["pressure"]) <= R["C3"]["pressure_abs_confirm"]
          and abs(c3["ball_recovery"]) <= R["C3"]["br_abs_confirm"] and abs(m3) <= R["C3"]["mean3_abs_confirm"]):
        J["C3"] = "确认"
    elif (abs(c3["pass"]) > R["C3"]["pass_abs_fail"] or abs(c3["pressure"]) > R["C3"]["pressure_abs_fail"]
          or abs(c3["ball_recovery"]) > R["C3"]["br_abs_fail"] or abs(m3) > R["C3"]["mean3_abs_fail"]):
        J["C3"] = "未确认"
    else:
        J["C3"] = "不确定"
    c4 = D["C4"]
    u1, sh = c4.get("u1", {}), c4.get("shuf_dest", {})
    if len(u1) < 4 or len(sh) < 3:
        J["C4"] = f"缺数据（u1 {len(u1)} 个种子，shuf_dest {len(sh)} 个种子）"
    elif (all(v["dest_rank"] == R["C4"]["u1_dest_rank"] and v["dest_gate"] >= R["C4"]["u1_dest_gate_min"] for v in u1.values())
          and all(v["dest_rank"] == R["C4"]["shuf_dest_rank"] and v["dest_gate"] <= R["C4"]["shuf_dest_gate_max"] for v in sh.values())):
        J["C4"] = "保持"
    elif any(v["dest_rank"] <= R["C4"]["shuf_fail_rank_at_most"] for v in sh.values()) or any(v["dest_rank"] != 1 for v in u1.values()):
        J["C4"] = "不保持"
    else:
        J["C4"] = "不确定"
    return J


def bootstrap(runs, n_boot):
    """按比赛重抽样，返回每个派生量和每个 (group, model, task) 主指标均值的 2.5、97.5 分位。"""
    matches = np.unique(np.concatenate([r["match"] for r in runs]))
    rng = np.random.RandomState(BOOT_SEED)
    idx = {m: i for i, m in enumerate(matches)}
    for r in runs:
        r["_mi"] = np.array([idx[m] for m in r["match"]])
    coll_D, coll_A = [], []
    t0 = time.time()
    for b in range(n_boot):
        counts = np.bincount(rng.randint(0, len(matches), len(matches)), minlength=len(matches)).astype(np.float64)
        agg_b = aggregate(runs, None)  # 占位，下面逐运行用权重重算
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
    """24/25 的参照值（冻结方案第二节抄来的，只用于并排显示）。"""
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
    L = [f"# 留出数据确认，25/26 的 {'146 场' if mode != 'check_2425' else '核对模式（24/25 的 202 场）'}汇总", "",
         f"由 `scripts/eval/holdout_2526_summarize.py` 生成，{time.strftime('%Y-%m-%d %H:%M')}。阈值字典原样如下，和冻结方案第二节逐字对应。", "",
         "```", json.dumps(RULES, ensure_ascii=False, indent=1), "```", ""]
    if mode == "check_2425":
        L += ["核对模式，读的是 24/25 的原预测限制到 202 场，用来和 `audit/u1_clean_summary.md` 对数，不判定。", ""]
    else:
        L += ["## 判定", "", "| 结论 | 判定 |", "|---|---|"] + [f"| {k} | **{v}** |" for k, v in J.items()] + [""]
    ref = ref_2425()
    c1 = D["C1"]
    L += ["## C1  落点不可替代", "", "| 量 | 留出集 | 95% 区间 | 24/25 |", "|---|---:|---|---:|",
          f"| u1fcn 三种子均值 top-1 | {fmt(c1['fcn_mean'])} | {fmt_ci(ci['derived']['C1']['fcn_mean']) if ci else ''} | {ref['C1']['fcn_mean']} |",
          f"| u1fcn 逐种子 | {' '.join(fmt(v) for v in c1['fcn_by_seed'].values())} | | 0.215 0.217 0.221 |",
          f"| 最强规则（{c1['best_rule']}） | {fmt(c1['best_rule_top1'])} | {fmt_ci(ci['derived']['C1']['best_rule_top1']) if ci else ''} | {ref['C1']['best_rule_top1']} |",
          f"| M2 加厚 | {fmt(c1['m2rich_mean'])} | {fmt_ci(ci['derived']['C1']['m2rich_mean']) if ci else ''} | {ref['C1']['m2rich_mean']} |",
          f"| M2 薄 | {fmt(c1['m2thin_mean'])} | {fmt_ci(ci['derived']['C1']['m2thin_mean']) if ci else ''} | {ref['C1']['m2thin_mean']} |",
          f"| u1fcn 减最强规则 | {fmt(c1['fcn_minus_rule'])} | {fmt_ci(ci['derived']['C1']['fcn_minus_rule']) if ci else ''} | 0.0599 |",
          f"| u1fcn 减 M2 加厚 | {fmt(c1['fcn_minus_m2rich'])} | {fmt_ci(ci['derived']['C1']['fcn_minus_m2rich']) if ci else ''} | 0.1173 |", ""]
    c2 = D["C2"]
    L += ["## C2  加厚标量替代大部分（判定用三任务平均，Pass、Ball Recovery、Pressure）", "", "| 量 | " + " | ".join(OUTCOME5) + " | 三任务平均 | 五任务平均 |", "|---|" + "---:|" * 7]
    for kk, label in (("thin_gain", "薄增益 M4 减 M2"), ("rich_gain", "厚增益 M4 减 M2"), ("xgbrich_minus_m4thin", "XGB 加厚减 M4 薄"), ("m4rich_minus_xgbrich", "M4 加厚减 XGB 加厚"),
                      ("m4thin_minus_xgbthin", "M4 薄减 XGB 薄（只报）"), ("xgbrich_minus_xgbthin", "XGB 加厚减 XGB 薄（只报）")):
        L.append(f"| {label} 留出集 | " + " | ".join(fmt(c2[kk].get(t)) for t in OUTCOME5) + f" | {fmt(c2[kk + '_mean3'])} | {fmt(c2[kk + '_mean5'])} |")
        if ci:
            L.append(f"| 同上 95% 区间 | " + " | ".join(fmt_ci(ci['derived']['C2'][kk][t]) for t in OUTCOME5) + f" | {fmt_ci(ci['derived']['C2'][kk + '_mean3'])} | {fmt_ci(ci['derived']['C2'][kk + '_mean5'])} |")
        L.append(f"| 同上 24/25 | " + " | ".join(fmt(ref['C2'][kk][t]) for t in OUTCOME5) + f" | {fmt(np.mean([ref['C2'][kk][t] for t in MEAN3]))} | {fmt(np.mean(list(ref['C2'][kk].values())))} |")
    L += [f"| 厚增益平均 对 薄增益平均 | | | | | | {fmt(c2['rich_over_thin_ratio3'], 2)}（24/25 是 0.25） | {fmt(c2['rich_over_thin_ratio5'], 2)}（24/25 是 0.43） |", ""]
    c3 = D["C3"]
    L += ["## C3  统一模型持平加厚 XGBoost（判定用 Pass、Pressure、Ball Recovery 和三任务平均）", "", "| 量 | " + " | ".join(OUTCOME5) + " | 五任务平均 | 三任务平均（Pass BR Pressure） |", "|---|" + "---:|" * 7]
    for kk, label in (("u1_minus_xgbrich", "u1 减 XGB 加厚"), ("u1rich_minus_xgbrich", "u1rich 减 XGB 加厚"), ("u1fcn_minus_u1", "u1fcn 减 u1"), ("u1rich_minus_u1", "u1rich 减 u1"), ("u1concat_minus_u1", "u1concat 减 u1")):
        m5 = fmt(c3["u1_minus_xgbrich_mean5"]) if kk == "u1_minus_xgbrich" else ""
        m3 = fmt(c3["u1_minus_xgbrich_mean3"]) if kk == "u1_minus_xgbrich" else ""
        L.append(f"| {label} 留出集 | " + " | ".join(fmt(c3[kk].get(t)) for t in OUTCOME5) + f" | {m5} | {m3} |")
        if ci:
            L.append(f"| 同上 95% 区间 | " + " | ".join(fmt_ci(ci['derived']['C3'][kk][t]) for t in OUTCOME5)
                     + (f" | {fmt_ci(ci['derived']['C3']['u1_minus_xgbrich_mean5'])} | {fmt_ci(ci['derived']['C3']['u1_minus_xgbrich_mean3'])} |" if kk == "u1_minus_xgbrich" else " | | |"))
        if kk in ref["C3"]:
            L.append(f"| 同上 24/25 | " + " | ".join(fmt(ref['C3'][kk][t]) for t in OUTCOME5) + f" | {fmt(np.mean(list(ref['C3'][kk].values())))} | {fmt(np.mean([ref['C3'][kk][t] for t in MEAN3]))} |")
    L.append("")
    L += ["## C4  落点门控读数的格局（不是机制的留出确认）", "", "| 运行 | 种子 | 落点门控 | 落点名次（六个任务里） | 各任务门控 |", "|---|---:|---:|---:|---|"]
    for tag, per in D["C4"].items():
        for sd, v in per.items():
            L.append(f"| {tag} | {sd} | {fmt(v['dest_gate'], 3)} | {v['dest_rank']} | " + " ".join(f"{t} {fmt(v['gates'][t], 3)}" for t in TASKS6) + " |")
    L += ["", "24/25 参照，u1 落点门控 0.78 到 0.83 四个种子都第 1；shuf_dest 0.24 到 0.28 三个种子都最后。", ""]
    # 全部模型的指标表
    ref_m = json.load(open(REF_JSON)) if (mode != "check_2425" and REF_JSON.exists()) else {}
    L += ["## 全部模型逐任务（种子均值 ± 标准差，n 种子；留出集；方括号是 24/25 的同一量和留出集减 24/25）", "", "| 组 | 模型 | " + " | ".join(TASKS6) + " |", "|---|---|" + "---:|" * 6]
    keys = sorted({(g, m) for (g, m, t) in agg}, key=lambda x: (x[0], x[1]))
    for g, m in keys:
        cells = []
        for t in TASKS6:
            v = agg.get((g, m, t), {})
            mm = key_metric(t)
            if mm in v:
                s = f"{fmt(v[mm]['mean'])} ± {fmt(v[mm]['sd'])} n={v[mm]['n']}"
                if t == "dest" and "skill" in v:
                    s += f"（skill {fmt(v['skill']['mean'])}）"
                if ci and "|".join((g, m, t)) in ci["metrics"]:
                    s += f" {fmt_ci(ci['metrics']['|'.join((g, m, t))])}"
                rk = "|".join((g, m, t))
                if rk in ref_m and mm in ref_m[rk]:
                    s += f" [24/25 {fmt(ref_m[rk][mm])}，变化 {fmt(v[mm]['mean'] - ref_m[rk][mm])}]"
                cells.append(s)
            else:
                cells.append("")
        L.append(f"| {g} | {m} | " + " | ".join(cells) + " |")
    L += ["", "落点一列是 top-1（括号里 skill），其余是 AUC。区间是按比赛重抽样的 95% 自助法区间（" + (f"{n_boot} 次，RandomState {BOOT_SEED}" if ci else "本次没做") + "）。绝对水平的变化是预期内的，不计入判定。", ""]
    # 按联赛拆
    L += ["## 按联赛拆（主指标，种子均值）", "", "| 组 | 模型 | 任务 | 西甲 | 英超 | 西甲条数 | 英超条数 |", "|---|---|---|---:|---:|---:|---:|"]
    byleague = {}
    for r in runs:
        lg = r.get("league")
        if lg is None:
            continue
        for L_name in ("西甲", "英超"):
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
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {fmt(np.nanmean(v.get('西甲', [np.nan])))} | {fmt(np.nanmean(v.get('英超', [np.nan])))} | {v.get('西甲_n', '')} | {v.get('英超_n', '')} |")
    L.append("")
    # 条数和标签
    L += ["## 留出集条数和标签均值（按 unified u1 第一个种子的事件表）", "", "| 任务 | 条数 | 标签均值 |", "|---|---:|---:|"]
    seen = set()
    for r in runs:
        if r["group"] == "unified" and r["model"] == "u1" and r["task"] not in seen:
            seen.add(r["task"])
            L.append(f"| {r['task']} | {len(r['y']):,} | {fmt(float(np.mean(r['y'])), 3)} |")
    L.append("")
    out_md = OUT_MD if mode != "check_2425" else Path("audit/holdout_2526_summary_check2425.md")
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
    print(f"[load] {len(runs)} 个运行×任务  ({time.time() - t0:.0f}s)", flush=True)
    if mode != "check_2425":
        lg = pd.read_parquet(HOLD / "match_table_2526.parquet", columns=["match_id", "season_dir"])
        lgmap = {int(m): ("西甲" if s.startswith("11_") else "英超") for m, s in zip(lg["match_id"], lg["season_dir"])}
        for r in runs:
            r["league"] = np.array([lgmap[int(m)] for m in r["match"]])
    agg = aggregate(runs)
    D = derived(agg)
    J = judge(D) if mode != "check_2425" else {}
    ci = bootstrap(runs, a.n_boot) if (mode != "check_2425" and a.n_boot > 0) else None
    if mode != "check_2425":
        out = dict(when=time.strftime("%Y-%m-%d %H:%M"), rules=RULES, judgement=J, derived=json.loads(json.dumps(D, default=lambda o: None)),
                   metrics={"|".join(k): {m: dict(mean=v["mean"], sd=v["sd"], n=v["n"], by_seed={str(s): x for s, x in v["by_seed"].items()}) for m, v in ms.items()} for k, ms in agg.items()},
                   ci=ci, n_boot=a.n_boot)
        json.dump(out, open(HOLD / "summary.json", "w"), ensure_ascii=False, indent=1)
    if mode == "check_2425":
        json.dump({"|".join(k): {m: v["mean"] for m, v in ms.items()} for k, ms in agg.items()}, open(REF_JSON, "w"), ensure_ascii=False, indent=1)
        print(f"[ref] 24/25 的各模型均值存到 {REF_JSON}", flush=True)
    write_md(runs, agg, D, J, ci, mode, a.n_boot)
    for k, v in J.items():
        print(f"[judge] {k}  {v}", flush=True)
    for k in sorted(agg):
        mm = key_metric(k[2])
        if mm in agg[k]:
            print(f"[agg] {k[0]:12s} {k[1]:36s} {k[2]:14s} {mm} {agg[k][mm]['mean']:.4f} ± {fmt(agg[k][mm]['sd'])} n={agg[k][mm]['n']}")
    print(f"[done] {OUT_MD if mode != 'check_2425' else 'audit/holdout_2526_summary_check2425.md'}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
