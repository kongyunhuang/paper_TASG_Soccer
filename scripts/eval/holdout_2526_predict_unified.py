#!/usr/bin/env python3
"""
留出数据确认，统一模型推理  用已保存的权重先复现 24/25 的逐事件预测，再（第二阶段）在 25/26 留出集上出预测
============================
对每个 {tag}_s{seed} 做三步。
  一  载入 .pt（权重和 scaler）和同名 json（fusion、encoder、channels、rich_scalars、hard_gate、shuffled）。
  二  复现  用 train_unified_clean.build_dataset 组出原测试集（八个任务，抽样固定），照训练脚本的顺序处理
      （加厚标量、按 RandomState(777) 先 dev 后 test 的打乱、两通道截取、scaler），预测后和原 npz 逐事件比，
      要求 task、y、event_id 完全相同，logit_bin、dest_logp_true、gate_mean 逐元素 |差| 不超过 max(1e-6, 2.4e-7×|值|)，
      也就是两个 float32 ulp 以内（跨机器的差是一两个 ulp，落点对数概率到 -15 时一个 ulp 就有 1.9e-6）。两种口径的超限个数都记下。复现不过就不往下做。
  三  留出  （加 --holdout 才做，第一阶段禁止）用 holdout_2526_dataset.build_holdout 组 25/26 留出集（六个任务，全量），
      同样处理后预测，存 task、y、logit_bin、dest_logp_true、dest_top1、gate_mean、match_id、event_id。
      打乱过的任务（shuffled 里列出的）在留出集内用 RandomState(777) 重新打乱一次，和训练时的干预一致。
不改 u1_clean 里的任何文件。权重目录可指定（重训的 u1 s1、s2 在 data/cache/holdout_2526/retrain/）。

输入  {weights_dir}/{tag}_s{seed}.pt 和 .json，data/cache/u1_clean/{tag}_s{seed}.npz（复现对照）
输出  data/cache/holdout_2526/unified/{tag}_s{seed}_repro.json      复现核对结果
      data/cache/holdout_2526/unified/{tag}_s{seed}_holdout.npz     留出集逐事件预测（--holdout 时）

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/eval/holdout_2526_predict_unified.py --tag u1rich --seeds 0,1,2            （只复现）
  PYTHONPATH=. python -u scripts/eval/holdout_2526_predict_unified.py --tag u1 --seeds 1,2 --weights_dir data/cache/holdout_2526/retrain
  PYTHONPATH=. python -u scripts/eval/holdout_2526_predict_unified.py --tag u1rich --seeds 0,1,2 --holdout   （第二阶段）

Last modified 2026-10-02
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
TOL = 1e-6        # 绝对容差
REL_TOL = 2.4e-7  # 相对容差，约两个 float32 ulp


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
    """照训练脚本的顺序处理  加厚标量（打乱之前）、按任务打乱空间张量（rs 的调用顺序和训练脚本一致）、拼接各任务、两通道截取。
    parts 是要拼的部分名列表（原测试集是 ("dev","test") 里只取 test 但打乱要先消耗 dev 的那次 permutation）。"""
    tasks = [t for t in TASKS if t in data]
    if meta.get("rich_scalars"):
        for t in tasks:
            for part in data[t]:
                d = data[t][part]
                if "X" not in d:
                    continue  # dev 只是占位（只记条数）
                d["X"] = np.hstack([d["X"], rich_scalars(d["sm"], d["X"][:, 0], d["X"][:, 1])])
    for t in meta.get("shuffled", []):
        if t not in data:
            continue
        for part in data[t]:
            d = data[t][part]
            n = d["_n"] if "_n" in d else len(d["sm"])
            perm = rs.permutation(n)  # 和训练脚本一样，先 dev 后 test 各消耗一次
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
    """照 sklearn StandardScaler.transform 的做法在 float32 上原地减均值、再除标准差（两次各舍入一次）。
    先在 float64 里算完再转 float32 会差一两个 ulp，复现不到 1e-6。"""
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
        # 相对口径  float32 一个 ulp 在量级 |a| 上约 1.2e-7×|a|，两台机器的差是一两个 ulp，落点对数概率到 -15 时一个 ulp 就是 1.9e-6
        r[f"{k}_n_over_rel_tol"] = int(np.sum(d > np.maximum(TOL, REL_TOL * np.abs(np.nan_to_num(a)))))
    r["pass"] = bool(r["task_equal"] and r["y_equal"] and r["eid_equal"] and all(r[f"{k}_n_over_rel_tol"] == 0 for k in ("logit_bin", "dest_logp_true", "gate_mean")))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--weights_dir", default=str(U1))
    ap.add_argument("--holdout", action="store_true", help="第二阶段才加。在 25/26 留出集上出预测")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TASG_THREADS", "4")))
    OUT.mkdir(parents=True, exist_ok=True)
    wdir = Path(a.weights_dir)
    seeds = [int(s) for s in a.seeds.split(",")]
    t0 = time.time()
    data_orig = build_dataset()
    print(f"[data] 原测试集组好  ({time.time() - t0:.0f}s)", flush=True)
    # dev 只需要条数（给打乱的 permutation 消耗用），张量释放
    dev_n = {t: len(data_orig[t]["dev"]["y"]) for t in TASKS}
    for t in TASKS:
        data_orig[t]["dev"] = {"_n": dev_n[t], "sm": None}
    hold = build_holdout() if a.holdout else None
    for seed in seeds:
        meta = json.load(open(wdir / f"{a.tag}_s{seed}.json"))
        # 复现 24/25
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
            print(f"[stop] {a.tag} s{seed} 复现不过，不做留出集", flush=True)
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
