#!/usr/bin/env python3
"""
留出数据确认，落点规则基线  把 dest_rule_baselines.py 的表和参数原样搬到 25/26 留出集上
============================
导入 dest_rule_baselines 的 table、rule、scores、start_cell 和常量，表只用训练赛季（build_dataset 的 dest dev，10 万条抽样）建，
a、b 照原脚本在训练赛季留出的两成比赛上按对数损失挑（和 audit/dest_rule_baselines.md 的 a=2，b=0.5 应完全一致），
不看 24/25 也不看 25/26。
  一  复现  在 24/25 的 202 场干净比赛上重算五条规则，和 audit/dest_rule_baselines.md 里的数逐个比（四位小数相同）。
  二  留出  （--holdout 时）在 25/26 留出集的 dest 任务上算五条规则，存逐事件的 top-1 和真格对数概率，供汇总脚本做自助法。

输入  同 dest_rule_baselines.py，另读 data/cache/holdout_2526/clean_matches_2526.parquet
输出  data/cache/holdout_2526/dest_rules_repro.json
      data/cache/holdout_2526/dest_rules_holdout.npz 和 dest_rules_holdout.json（--holdout 时）

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/eval/holdout_2526_dest_rules.py            （只复现）
  PYTHONPATH=. python -u scripts/eval/holdout_2526_dest_rules.py --holdout  （第二阶段）

Last modified 2026-10-02
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
ORIG_MD = Path("audit/dest_rule_baselines.md")


def all_rules(T, pm, X, sm, best_a, best_ab):
    st = start_cell(X)
    flat = lambda c: np.asarray(sm[:, c]).reshape(len(sm), -1)
    mt, ot = flat(0), flat(1)
    n = len(X)
    return {"均匀分布": np.full((n, N_DEST), 1 / N_DEST), "边际分布": np.tile(pm, (n, 1)), "起点格条件表": T[st],
            f"条件表乘队友占位（a={best_a[1]}）": rule(T, st, mt, ot, best_a[1], 0),
            f"条件表乘队友再除对手（a={best_ab[1]}，b={best_ab[2]}）": rule(T, st, mt, ot, best_ab[1], best_ab[2])}


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

    # 复现 24/25
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
        print(f"[repro] {name:34s} top1 {s['top1']:.4f} logloss {s['logloss']:.4f}  原 {o}  {'相同' if same else '不同'}", flush=True)
    rep["_pass"] = bool(all_ok)
    rep["_n_test_202"] = int(ok.sum())
    rep["_params"] = dict(a_only=best_a[1], a=best_ab[1], b=best_ab[2])
    rep["_when"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(rep, open(OUT_DIR / "dest_rules_repro.json", "w"), ensure_ascii=False, indent=1)
    print(f"[repro] {'全部相同' if all_ok else '有不同'}", flush=True)
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
    res["_when"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(res, open(OUT_DIR / "dest_rules_holdout.json", "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(OUT_DIR / "dest_rules_holdout.npz", **save)
    print(f"[done] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
