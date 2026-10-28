#!/usr/bin/env python3
"""
train_single_clean.py, single-task models on the same per-task data as the unified model
========================================================================================
Measures the gain of the CNN and of the gate on each task separately. The data come directly from
train_unified_clean.build_dataset, so the events, features, labels and test set (229 normal-state matches)
of every task are identical to those of the unified model, and the single-task gains can be compared with
the unified gate.
Four models per task, with the architectures of train_all.py and an output of size 1 (binary) or 96 (pass destination).
  B2  XGBoost on event features plus 360 scalars (binary tasks only)
  M2  scalar MLP
  M4  CNN with concatenation
  G1  gated fusion
Each seed redraws the train and validation matches, and the four models share that split. Runs can be resumed.
With --rich_scalars, 15 hand-crafted summaries of the spatial tensor are added to the scalar features (same option
as in the unified model). This measures how much CNN gain remains once the scalar branch is richer. Those results
go to data/cache/single_clean_rich/.

Inputs   same as train_unified_clean.py
Outputs  data/cache/single_clean/{task}.jsonl
         data/cache/single_clean/preds/{task}_{model}_s{seed}.npz
         data/cache/single_clean/preds/{task}_test_meta.npz

Usage
  PYTHONPATH=. python -u scripts/training/train_single_clean.py --tasks dest,pressure --seeds 0,1,2,3,4
  PYTHONPATH=. python -u scripts/training/train_single_clean.py --rich_scalars --seeds 0,1,2
  PYTHONPATH=. python -u scripts/training/train_single_clean.py --smoke

Last modified 2026-10-28
"""

from __future__ import annotations

import argparse
import os
import socket
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from scripts.training.train_all import CACHE_DIR, SoccerMapEncoder
from scripts.training.train_unified_clean import N_DEST, TASKS, build_dataset, rich_scalars

OUT_DIR = CACHE_DIR / "single_clean"
PRED_DIR = OUT_DIR / "preds"
MODELS = ["B2_XGB_360", "M2_MLP_360", "M4_CNN_Full", "G1_Gating"]
SALT = {"B2_XGB_360": 1, "M2_MLP_360": 2, "M4_CNN_Full": 3, "G1_Gating": 4}


class MLP(nn.Module):
    def __init__(self, d, out, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ELU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.ELU(), nn.Dropout(dropout),
                                 nn.Linear(32, 16), nn.ELU(), nn.Dropout(dropout), nn.Linear(16, out))

    def forward(self, x, s):
        return self.net(x), None


class Concat(nn.Module):
    def __init__(self, d, out, e=64, dropout=0.3):
        super().__init__()
        self.event_mlp = nn.Sequential(nn.Linear(d, e), nn.ELU(), nn.Dropout(dropout))
        self.cnn = SoccerMapEncoder(7, e)
        self.head = nn.Sequential(nn.Linear(e * 2, 32), nn.ELU(), nn.Dropout(dropout), nn.Linear(32, out))

    def forward(self, x, s):
        return self.head(torch.cat([self.event_mlp(x), self.cnn(s)], -1)), None


class Gating(nn.Module):
    def __init__(self, d, out, e=64, dropout=0.3):
        super().__init__()
        self.event_enc = nn.Sequential(nn.Linear(d, e), nn.ELU(), nn.Dropout(dropout), nn.Linear(e, e), nn.Tanh())
        self.spatial_enc = SoccerMapEncoder(7, e)
        self.gate = nn.Sequential(nn.Linear(e * 2, e), nn.Sigmoid())
        self.pred = nn.Sequential(nn.Linear(e, 32), nn.ELU(), nn.Dropout(dropout), nn.Linear(32, out))

    def forward(self, x, s):
        h_e, h_s = self.event_enc(x), self.spatial_enc(s)
        g = self.gate(torch.cat([h_e, h_s], -1))
        return self.pred(g * h_s + (1 - g) * h_e), g


def forward_all(model, X, S, bs=8192):
    model.eval()
    outs, gates = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            o, g = model(X[i:i + bs], S[i:i + bs])
            outs.append(o)
            gates.append(g.mean(1) if g is not None else torch.full((len(o),), float("nan")))
    return torch.cat(outs), torch.cat(gates).numpy()


def score(out, y, multiclass):
    """Return (score used for model selection, metrics dict)."""
    if multiclass:
        lp = torch.log_softmax(out, -1).numpy()
        nll = float(-lp[np.arange(len(y)), y].mean())
        top1 = float((lp.argmax(1) == y).mean())
        top3 = float((np.argsort(-lp, axis=1)[:, :3] == y[:, None]).any(1).mean())
        return -nll, dict(logloss=nll, top1=top1, top3=top3, skill=1 - nll / math.log(N_DEST))
    auc = float(roc_auc_score(y, out.squeeze(-1).numpy()))
    return auc, dict(auc=auc, skill=auc)


def train_nn(model, A, multiclass, seed, epochs, patience=7):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss() if multiclass else nn.BCEWithLogitsLoss()
    n = len(A["Xtr"])
    best, state, bad = -1e9, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            o, _ = model(A["Xtr"][b], A["Str"][b])
            loss = lossf(o, A["Ytr"][b]) if multiclass else lossf(o.squeeze(-1), A["Ytr"][b].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        vo, _ = forward_all(model, A["Xva"], A["Sva"])
        s, _ = score(vo, A["yva"], multiclass)
        if s > best:
            best, bad = s, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rich_scalars", action="store_true", help="add 15 hand-crafted summaries of the spatial tensor to the scalar features; results go to single_clean_rich")
    a = ap.parse_args()
    global OUT_DIR, PRED_DIR
    if a.rich_scalars:
        OUT_DIR = CACHE_DIR / "single_clean_rich"
        PRED_DIR = OUT_DIR / "preds"
    torch.set_num_threads(int(os.environ.get("TASG_THREADS", "4")))  # number of threads from TASG_THREADS, default 4
    tasks = a.tasks.split(",")
    seeds = [0] if a.smoke else [int(s) for s in a.seeds.split(",")]
    models = a.models.split(",")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = build_dataset(a.smoke)
    if a.rich_scalars:
        for t in TASKS:
            for part in ("dev", "test"):
                d = data[t][part]
                d["X"] = np.hstack([d["X"], rich_scalars(d["sm"], d["X"][:, 0], d["X"][:, 1])])
        print(f"[rich] scalar features widened to {data[TASKS[0]]['dev']['X'].shape[1]}", flush=True)
    total = sum(len(seeds) * len([m for m in models if not (t == "dest" and m == "B2_XGB_360")]) for t in tasks)
    done = 0
    for task in tasks:
        multiclass = task == "dest"
        out_path = OUT_DIR / f"{task}.jsonl"
        have = set()
        if out_path.exists() and not a.smoke:
            have = {(json.loads(x)["model"], json.loads(x)["seed"]) for x in open(out_path)}
        dev, test = data[task]["dev"], data[task]["test"]
        if not a.smoke:
            np.savez_compressed(PRED_DIR / f"{task}_test_meta.npz", event_id=test["eid"], match_id=test["match"], y=test["y"].astype(np.int16))
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
                else:
                    torch.manual_seed(seed * 100003 + SALT[name])
                    out_dim = N_DEST if multiclass else 1
                    model = {"M2_MLP_360": MLP, "M4_CNN_Full": Concat, "G1_Gating": Gating}[name](Xtr.shape[1], out_dim)
                    model = train_nn(model, A, multiclass, seed, epochs=1 if a.smoke else 30)
                    vo, _ = forward_all(model, A["Xva"], A["Sva"])
                    to, gate = forward_all(model, Xte_t, Ste)
                    _, vm = score(vo, A["yva"], multiclass)
                    _, tm = score(to, test["y"], multiclass)
                    if multiclass:
                        lp = torch.log_softmax(to, -1).numpy()
                        save = dict(logp_true=lp[np.arange(len(test["y"])), test["y"]].astype(np.float32), top1=lp.argmax(1).astype(np.int16))
                    else:
                        save = dict(prob=(1 / (1 + np.exp(-to.squeeze(-1).numpy()))).astype(np.float32))
                    if name == "G1_Gating":
                        save["gate"] = gate.astype(np.float32)
                res = dict(task=task, model=name, seed=seed, test=tm, val=vm, time_s=round(time.time() - t1, 1), host=socket.gethostname())
                if name == "G1_Gating":
                    res["gate_mean"], res["gate_event_sd"] = float(np.mean(gate)), float(np.std(gate))
                done += 1
                el = time.time() - t0
                main_metric = f"top1={tm['top1']:.3f} nll={tm['logloss']:.3f}" if multiclass else f"auc={tm['auc']:.4f}"
                print(f"[{done}/{total}] {task} {name} seed={seed} test {main_metric} "
                      f"{'gate=' + format(res['gate_mean'], '.3f') if 'gate_mean' in res else ''} ({res['time_s']:.0f}s) "
                      f"elapsed {el / 60:.1f} min  eta {el / done * (total - done) / 60:.0f} min", flush=True)
                if not a.smoke:
                    with open(out_path, "a") as f:
                        f.write(json.dumps(res) + "\n")
                    np.savez_compressed(PRED_DIR / f"{task}_{name}_s{seed}.npz", **save)
    print(f"[done] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
