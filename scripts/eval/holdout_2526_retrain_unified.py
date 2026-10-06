#!/usr/bin/env python3
"""
留出数据确认，补训统一模型权重  用原命令重跑 u1 的种子 1、2，输出改到 holdout_2526/retrain/，不碰 u1_clean
============================
基准 u1 只有种子 3 有权重（种子 0 用 u1chk_s0 代替）。train_unified_clean.py 见到同名 json 就跳过，换标签又会被
summarize_u1_clean.py 当成新运行写进权威汇总表，所以这里导入 train_unified_clean，把它的 OUT_DIR 指到
data/cache/holdout_2526/retrain/，再用原参数（--seed N --tag u1，其余全默认）调用 main()。训练过程一行不改。
  --verify  训练完后和 data/cache/u1_clean/u1_s{seed}.json、.npz 核对。口径
            json 里八个任务的 test、val、gate_mean 和 history（逐轮 train_loss、val_score、val_gate）、best_val_score 完全相等；
            npz 逐事件 task、y、event_id 相同，logit_bin、dest_logp_true、gate_mean 逐元素 |差| 不超过 max(1e-6, 2.4e-7×|值|)。
            结果写 data/cache/holdout_2526/retrain/u1_s{seed}_verify.json。

输入  同 train_unified_clean.py
输出  data/cache/holdout_2526/retrain/u1_s{seed}.{json,pt,npz}，u1_s{seed}_verify.json

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/eval/holdout_2526_retrain_unified.py --seed 1 --verify      （Mac mini，约 16 分钟）
  PYTHONPATH=. python -u scripts/eval/holdout_2526_retrain_unified.py --seed 1 --verify_only  （只核对，不训练）

Last modified 2026-10-03
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
    # history 只比两边都有的字段。原 u1 s1、s2 是早上的脚本训的，每轮只记 epoch、train_loss、val_score；晚上的脚本多记了 val_gate
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
        tuc.OUT_DIR = RETRAIN_DIR  # main() 里所有输出路径都从这个模块变量取
        sys.argv = ["train_unified_clean.py", "--seed", str(a.seed), "--tag", a.tag]
        print(f"[retrain] {a.tag} s{a.seed} -> {RETRAIN_DIR}  argv {sys.argv[1:]}", flush=True)
        tuc.main()
    if a.verify or a.verify_only:
        verify(a.seed, a.tag)


if __name__ == "__main__":
    main()
