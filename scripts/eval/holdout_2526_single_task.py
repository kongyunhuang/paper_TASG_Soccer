#!/usr/bin/env python3
"""
留出数据确认，单任务模型  照 train_single_clean 的流程重训并保存权重，先复现 24/25 的预测，再（第二阶段）用权重给 25/26 留出集出预测
============================
单任务模型今天只存了 24/25 的预测，没存权重。本脚本导入 train_single_clean 的模型类、train_nn、score、forward_all、SALT，
以及 train_unified_clean 的 build_dataset 和 rich_scalars，按原脚本 main() 的顺序逐任务、逐种子、逐模型训练（同样的划分、
scaler、manual_seed、轮数、早停），训练过程一行不改，训练完多做两件事，把 24/25 的预测和 data/cache/single_clean{_rich}/preds/
里的原文件逐元素比（差超过 max(1e-6, 2.4e-7×|值|) 记为复现不过），并把权重和 scaler 存下来。
第二阶段用 --predict_holdout 从权重出 25/26 的预测，训练和留出预测分开，训练可以在方案定稿前先跑，不碰 25/26。
输出全部写到 data/cache/holdout_2526/single{_rich}/，不碰 single_clean、single_clean_rich、u1_clean。
  --models  默认只跑 M2_MLP_360 和 M4_CNN_Full（判定要用的）。B2_XGB_360 只在 MacBook 上跑，神经网络只在 Mac mini 上跑。
可断点续跑（按 {task}.jsonl 里已有的 model、seed 跳过）。

输入  同 train_single_clean.py，另读 data/cache/single_clean{_rich}/preds/*.npz（复现对照）
输出  data/cache/holdout_2526/single{_rich}/{task}.jsonl                       指标（和原格式相同，另加 repro 字段）
      data/cache/holdout_2526/single{_rich}/preds/{task}_{model}_s{seed}.npz     24/25 预测（和原格式相同）
      data/cache/holdout_2526/single{_rich}/weights/{task}_{model}_s{seed}.pt    神经网络 state_dict 加 scaler；XGBoost 存 .ubj 加 _scaler.npz
      data/cache/holdout_2526/single{_rich}/holdout_preds/{task}_{model}_s{seed}.npz   留出集预测（--predict_holdout 时）
      data/cache/holdout_2526/single{_rich}/holdout_preds/{task}_holdout_meta.npz      留出集 event_id、match_id、y
      data/cache/holdout_2526/single{_rich}/holdout_preds/{task}_{model}_s{seed}_repro.json   从权重重算 24/25 和训练时预测的差

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/eval/holdout_2526_single_task.py --models B2_XGB_360 --seeds 0,1,2,3,4      （MacBook，训练并复现）
  PYTHONPATH=. python -u scripts/eval/holdout_2526_single_task.py --seeds 0,1,2,3,4                            （Mac mini，M2 M4 薄标量）
  PYTHONPATH=. python -u scripts/eval/holdout_2526_single_task.py --rich_scalars --seeds 0,1,2                 （Mac mini，M2 M4 加厚）
  PYTHONPATH=. python -u scripts/eval/holdout_2526_single_task.py --recheck_repro --models M2_MLP_360,M4_CNN_Full --seeds 0,1,2,3,4   （同步回 MacBook 后补核复现）
  PYTHONPATH=. python -u scripts/eval/holdout_2526_single_task.py --predict_holdout --models B2_XGB_360,M2_MLP_360,M4_CNN_Full --seeds 0,1,2,3,4   （第二阶段）

Last modified 2026-10-02
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from scripts.eval.holdout_2526_dataset import HOLDOUT_TASKS, build_holdout
from scripts.training.train_all import CACHE_DIR
from scripts.training.train_single_clean import MLP, SALT, Concat, Gating, forward_all, score, train_nn
from scripts.training.train_unified_clean import N_DEST, TASKS, build_dataset, rich_scalars

TOL, REL_TOL = 1e-6, 2.4e-7
NN = {"M2_MLP_360": MLP, "M4_CNN_Full": Concat, "G1_Gating": Gating}


def standardize(X, mean, scale):
    """照 sklearn StandardScaler.transform 在 float32 上原地减均值再除标准差，和训练时逐位一致。"""
    Z = np.array(X, dtype=np.float32, copy=True)
    Z -= mean
    Z /= scale
    return Z


def compare_pred(save, orig_path):
    """和原预测文件逐元素比。返回 dict(pass_, 每个字段的最大绝对差和超限个数)。"""
    if not orig_path.exists():
        return dict(orig_exists=False, pass_=False)
    z = np.load(orig_path, allow_pickle=True)
    r = dict(orig_exists=True)
    ok = True
    for k, v in save.items():
        if k not in z.files:
            r[f"{k}_missing_in_orig"] = True
            ok = False
            continue
        a, b = z[k].astype(np.float64), np.asarray(v).astype(np.float64)
        if a.shape != b.shape:
            r[f"{k}_shape"] = f"{a.shape} vs {b.shape}"
            ok = False
            continue
        d = np.abs(a - b)
        r[f"{k}_max_abs_diff"] = float(d.max()) if d.size else 0.0
        r[f"{k}_n_over_abs_tol"] = int((d > TOL).sum())
        r[f"{k}_n_over_rel_tol"] = int((d > np.maximum(TOL, REL_TOL * np.abs(a))).sum())
        ok &= r[f"{k}_n_over_rel_tol"] == 0
    r["pass_"] = bool(ok)
    return r


def nn_outputs(model, X_t, S_t, y, multiclass, name):
    o, gate = forward_all(model, X_t, S_t)
    if multiclass:
        lp = torch.log_softmax(o, -1).numpy()
        out = dict(logp_true=lp[np.arange(len(y)), y].astype(np.float32), top1=lp.argmax(1).astype(np.int16))
    else:
        out = dict(prob=(1 / (1 + np.exp(-o.squeeze(-1).numpy()))).astype(np.float32))
    if name == "G1_Gating":
        out["gate"] = gate.astype(np.float32)
    return out, o, gate


def train_mode(a, dirs, tasks, seeds, models, orig_dir):
    out_dir, pred_dir, w_dir = dirs["out"], dirs["pred"], dirs["w"]
    t0 = time.time()
    data = build_dataset()  # 和原脚本一样先组全部八个任务（抽样的随机数序列才相同）
    if a.rich_scalars:
        for t in TASKS:
            for part in ("dev", "test"):
                d = data[t][part]
                d["X"] = np.hstack([d["X"], rich_scalars(d["sm"], d["X"][:, 0], d["X"][:, 1])])
        print(f"[rich] scalar features widened to {data[TASKS[0]]['dev']['X'].shape[1]}", flush=True)
    for t in TASKS:
        if t not in tasks:
            del data[t]  # 不用的任务释放内存（随机数已经消耗完，不影响抽样）
    total = sum(len(seeds) * len([m for m in models if not (t == "dest" and m == "B2_XGB_360")]) for t in tasks)
    done = 0
    for task in tasks:
        multiclass = task == "dest"
        out_path = out_dir / f"{task}.jsonl"
        have = set()
        if out_path.exists():
            have = {(json.loads(x)["model"], json.loads(x)["seed"]) for x in open(out_path)}
        dev, test = data[task]["dev"], data[task]["test"]
        np.savez_compressed(pred_dir / f"{task}_test_meta.npz", event_id=test["eid"], match_id=test["match"], y=test["y"].astype(np.int16))
        matches = np.sort(np.unique(dev["match"]))
        for seed in seeds:
            perm = np.random.RandomState(20261002 + seed).permutation(len(matches))
            tr = np.isin(dev["match"], matches[perm[:int(0.8 * len(matches))]])
            va = ~tr
            sc = StandardScaler().fit(dev["X"][tr])
            Xtr, Xva, Xte = sc.transform(dev["X"][tr]), sc.transform(dev["X"][va]), sc.transform(test["X"])
            A = dict(Xtr=torch.tensor(Xtr, dtype=torch.float32), Str=torch.tensor(dev["sm"][tr]), Ytr=torch.tensor(dev["y"][tr], dtype=torch.long),
                     Xva=torch.tensor(Xva, dtype=torch.float32), Sva=torch.tensor(dev["sm"][va]), yva=dev["y"][va])
            Xte_t, Ste = torch.tensor(Xte, dtype=torch.float32), torch.tensor(test["sm"])
            for name in models:
                if (multiclass and name == "B2_XGB_360") or (name, seed) in have:
                    continue
                t1 = time.time()
                gate = None
                if name == "B2_XGB_360":
                    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                        min_child_weight=5, eval_metric="auc", tree_method="hist", random_state=seed, verbosity=0)
                    xgb.fit(Xtr, dev["y"][tr], eval_set=[(Xva, dev["y"][va])], verbose=False)
                    vm = dict(auc=float(roc_auc_score(dev["y"][va], xgb.predict_proba(Xva)[:, 1])))
                    p = xgb.predict_proba(Xte)[:, 1]
                    tm = dict(auc=float(roc_auc_score(test["y"], p)))
                    save = dict(prob=p.astype(np.float32))
                    xgb.save_model(w_dir / f"{task}_{name}_s{seed}.ubj")
                    np.savez(w_dir / f"{task}_{name}_s{seed}_scaler.npz", mean=sc.mean_, scale=sc.scale_)
                else:
                    torch.manual_seed(seed * 100003 + SALT[name])
                    out_dim = N_DEST if multiclass else 1
                    model = NN[name](Xtr.shape[1], out_dim)
                    model = train_nn(model, A, multiclass, seed, epochs=30)
                    vo, _ = forward_all(model, A["Xva"], A["Sva"])
                    _, vm = score(vo, A["yva"], multiclass)
                    save, to, gate = nn_outputs(model, Xte_t, Ste, test["y"], multiclass, name)
                    _, tm = score(to, test["y"], multiclass)
                    torch.save({"state": model.state_dict(), "scaler_mean": sc.mean_, "scaler_scale": sc.scale_, "n_feat": int(Xtr.shape[1]), "out_dim": out_dim},
                               w_dir / f"{task}_{name}_s{seed}.pt")
                rep = compare_pred(save, orig_dir / "preds" / f"{task}_{name}_s{seed}.npz")
                res = dict(task=task, model=name, seed=seed, test=tm, val=vm, time_s=round(time.time() - t1, 1), host=socket.gethostname(), repro=rep)
                if name == "G1_Gating":
                    res["gate_mean"], res["gate_event_sd"] = float(np.mean(gate)), float(np.std(gate))
                done += 1
                el = time.time() - t0
                main_metric = f"top1={tm['top1']:.3f} nll={tm['logloss']:.3f}" if multiclass else f"auc={tm['auc']:.4f}"
                print(f"[{done}/{total}] {task} {name} seed={seed} test {main_metric} repro={'pass' if rep['pass_'] else 'FAIL'} "
                      f"({res['time_s']:.0f}s) elapsed {el / 60:.1f} min  eta {el / done * (total - done) / 60:.0f} min", flush=True)
                with open(out_path, "a") as f:
                    f.write(json.dumps(res) + "\n")
                np.savez_compressed(pred_dir / f"{task}_{name}_s{seed}.npz", **save)
    print(f"[done] {(time.time() - t0) / 60:.1f} min", flush=True)


def predict_mode(a, dirs, tasks, seeds, models):
    """第二阶段。从权重出 25/26 留出集的预测；同时从权重重算 24/25 并和训练时存的预测比，确认权重没存错。"""
    pred_dir, w_dir, h_dir = dirs["pred"], dirs["w"], dirs["hold"]
    t0 = time.time()
    data = build_dataset()
    hold = build_holdout(tasks)
    if a.rich_scalars:
        for t in tasks:
            d = data[t]["test"]
            d["X"] = np.hstack([d["X"], rich_scalars(d["sm"], d["X"][:, 0], d["X"][:, 1])])
            hold[t]["X"] = np.hstack([hold[t]["X"], rich_scalars(hold[t]["sm"], hold[t]["X"][:, 0], hold[t]["X"][:, 1])])
    for t in TASKS:
        if t not in tasks:
            del data[t]
        else:
            del data[t]["dev"]
    n_done = 0
    for task in tasks:
        multiclass = task == "dest"
        test, h = data[task]["test"], hold[task]
        np.savez_compressed(h_dir / f"{task}_holdout_meta.npz", event_id=h["eid"], match_id=h["match"], y=h["y"].astype(np.int16))
        for seed in seeds:
            for name in models:
                if multiclass and name == "B2_XGB_360":
                    continue
                if name == "B2_XGB_360":
                    wp = w_dir / f"{task}_{name}_s{seed}.ubj"
                    if not wp.exists():
                        print(f"[miss] {wp}", flush=True)
                        continue
                    xgb = XGBClassifier()
                    xgb.load_model(wp)
                    scz = np.load(w_dir / f"{task}_{name}_s{seed}_scaler.npz")
                    mean, scale = scz["mean"], scz["scale"]
                    rep_save = dict(prob=xgb.predict_proba(standardize(test["X"], mean, scale))[:, 1].astype(np.float32))
                    hsave = dict(prob=xgb.predict_proba(standardize(h["X"], mean, scale))[:, 1].astype(np.float32))
                else:
                    wp = w_dir / f"{task}_{name}_s{seed}.pt"
                    if not wp.exists():
                        print(f"[miss] {wp}", flush=True)
                        continue
                    ck = torch.load(wp, weights_only=False)
                    model = NN[name](ck["n_feat"], ck["out_dim"])
                    model.load_state_dict(ck["state"])
                    mean, scale = ck["scaler_mean"], ck["scaler_scale"]
                    rep_save, _, _ = nn_outputs(model, torch.tensor(standardize(test["X"], mean, scale)), torch.tensor(test["sm"]), test["y"], multiclass, name)
                    hsave, _, _ = nn_outputs(model, torch.tensor(standardize(h["X"], mean, scale)), torch.tensor(h["sm"]), h["y"], multiclass, name)
                rep = compare_pred(rep_save, pred_dir / f"{task}_{name}_s{seed}.npz")
                rep.update(task=task, model=name, seed=seed, host=socket.gethostname(), when=time.strftime("%Y-%m-%d %H:%M"))
                json.dump(rep, open(h_dir / f"{task}_{name}_s{seed}_repro.json", "w"), ensure_ascii=False, indent=1)
                if not rep["pass_"]:
                    print(f"[stop] {task} {name} s{seed} 从权重重算的 24/25 预测和训练时不同，不出留出集预测  {rep}", flush=True)
                    continue
                np.savez_compressed(h_dir / f"{task}_{name}_s{seed}.npz", **hsave)
                n_done += 1
                print(f"[holdout] {task} {name} s{seed} n {len(h['y']):,} saved  repro max diff {max(v for k, v in rep.items() if k.endswith('max_abs_diff')):.2e}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"[done] {n_done} 个留出预测  {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(HOLDOUT_TASKS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--models", default="M2_MLP_360,M4_CNN_Full")
    ap.add_argument("--rich_scalars", action="store_true")
    ap.add_argument("--predict_holdout", action="store_true", help="第二阶段才加。从已存权重给 25/26 留出集出预测")
    ap.add_argument("--recheck_repro", action="store_true", help="只重新比对 preds/ 和原文件（训练机上缺原文件时，同步回来后在 MacBook 上补核）")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TASG_THREADS", "4")))
    orig_dir = CACHE_DIR / ("single_clean_rich" if a.rich_scalars else "single_clean")
    out_dir = CACHE_DIR / "holdout_2526" / ("single_rich" if a.rich_scalars else "single")
    dirs = dict(out=out_dir, pred=out_dir / "preds", w=out_dir / "weights", hold=out_dir / "holdout_preds")
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    tasks = a.tasks.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    models = a.models.split(",")
    assert all(t in HOLDOUT_TASKS for t in tasks), "留出任务只有 pass dest shot interception ball_recovery pressure"
    if a.recheck_repro:
        rows = []
        for task in tasks:
            for seed in seeds:
                for name in models:
                    f = dirs["pred"] / f"{task}_{name}_s{seed}.npz"
                    if not f.exists():
                        continue
                    z = np.load(f, allow_pickle=True)
                    rep = compare_pred({k: z[k] for k in z.files}, orig_dir / "preds" / f"{task}_{name}_s{seed}.npz")
                    rep.update(task=task, model=name, seed=seed, when=time.strftime("%Y-%m-%d %H:%M"))
                    rows.append(rep)
                    print(f"[recheck] {task} {name} s{seed} pass={rep['pass_']} " + " ".join(f"{k}={rep[k]:.2e}" for k in rep if k.endswith("max_abs_diff")), flush=True)
        with open(dirs["out"] / "repro_recheck.jsonl", "w") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rows))
        print(f"[recheck] {len(rows)} 个，不通过 {sum(not r['pass_'] for r in rows)} 个", flush=True)
    elif a.predict_holdout:
        predict_mode(a, dirs, tasks, seeds, models)
    else:
        train_mode(a, dirs, tasks, seeds, models, orig_dir)


if __name__ == "__main__":
    main()
