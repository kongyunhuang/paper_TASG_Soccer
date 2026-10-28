#!/usr/bin/env python3
"""
train_unified_clean.py, unified multi-task gating model with the pass destination task
======================================================================================
Trains the unified gating model (event encoder, SoccerMap encoder, gate, task embedding) on the cleaned task set.
  Tasks      pass, dest (the 8x12 cell of the pass end location, 96 classes), shot, interception, ball_recovery,
             pressure, dribble, tackle. Tackle is the only Duel type kept.
  Data       dribble and tackle use the re-oriented tensors and recomputed 360 scalars (data/cache/perspective_fixed/).
             The pressure label uses only team and time. A pressure succeeds when, within 5 s of its start and in the
             same half, the pressing team completes a Pass, Carry, Shot or Dribble.
             The test set uses only the 229 matches of 24/25 whose 360 frames are in the normal state; the 531 matches
             in the reversed state are excluded entirely.
  Features   duration and dribble_overrun are not used (they are known only after the action ends). Pass end point,
             length, angle and the other pass features are given to the pass task only; for dest these dimensions are zero.
  Heads      the binary tasks share one 1-d output and dest has a 96-d output; both read the same fused representation.
  Loss       BCE for the binary tasks, cross-entropy times 1/ln(96) for dest, so that both losses have the same scale
             under a uniform guess.
  Ablation   after training, the test spatial tensors are shuffled within each task and the model predicts again. The
             drop of each task metric (spatial_reliance) is a direct, gate-independent measure of how much spatial
             information the model uses on that task. The model weights are saved as well.
  Split      the test set and the per-task subsamples are fixed (numpy seed 42); each seed redraws the train and
             validation matches and changes the initialisation and batch order.
Encoders
  --encoder pooled  SoccerMap encoder pooled to 2x2 and flattened; dest gets 96 logits from the fused vector.
  --encoder fcn     keeps the 8x12 resolution. The spatial branch outputs a per-cell feature map, the scalar branch
                    broadcasts its vector to every cell and appends the cell coordinates, and the same gate vector mixes
                    the two in every cell; dest gets one logit per cell. The binary tasks are unchanged (the feature map
                    is pooled to a vector before gated fusion). Spatial detail for dest is thus not lost in pooling, the
                    scalar branch can still give a location prior for dest on its own, and the gate still gives the share
                    of each hidden dimension taken from the spatial branch.
Interventions
  --shuffle_spatial  shuffles the spatial tensors of the named tasks across events of the same task, so the spatial
                     input carries no information for that task; tests whether the gate closes in response.
  --fusion concat    replaces the gate with concatenation followed by a linear map, as a control within the same model.

Inputs   data/cache/L1_events_v3.parquet
         data/cache/action_soccermaps_{pass,shot,interception,ball_recovery,pressure}.npy and _idx.parquet
         data/cache/perspective_fixed/soccermaps_{dribble,duel}.npy and _idx.parquet
         data/raw_scan/match_state_2425.parquet
Outputs  data/cache/u1_clean/{tag}_s{seed}.json   per-task test metrics, validation metrics, gate means
         data/cache/u1_clean/{tag}_s{seed}.npz    per-event test predictions and gate means

Usage
  PYTHONPATH=. python -u scripts/training/train_unified_clean.py --seed 0
  PYTHONPATH=. python -u scripts/training/train_unified_clean.py --seed 0 --shuffle_spatial dest --tag shuf_dest
  PYTHONPATH=. python -u scripts/training/train_unified_clean.py --seed 0 --channels 2 --tag u1_2ch   (spatial branch gets only the two player position channels)
  PYTHONPATH=. python -u scripts/training/train_unified_clean.py --seed 0 --rich_scalars --tag u1rich   (scalar branch gets 15 hand-crafted summaries of the tensor)
  PYTHONPATH=. python -u scripts/training/train_unified_clean.py --seed 0 --gate_penalty 0.01 --tag u1pen010   (opening the gate has a cost, loss + 0.01 x gate mean)
  PYTHONPATH=. python -u scripts/training/train_unified_clean.py --seed 0 --hard_gate --gate_penalty 0.01 --tag u1hard010   (hard gate with the cost)
  PYTHONPATH=. python -u scripts/training/train_unified_clean.py --seed 0 --smoke   (a few thousand events per task, one epoch, only checks that it runs)

Last modified 2026-10-28
"""

from __future__ import annotations

import argparse
import os
import socket
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.training.train_all import CACHE_DIR, FULL_SEASONS, TRAIN_SEASONS, TEST_SEASONS

OUT_DIR = CACHE_DIR / "u1_clean"
FIX_DIR = CACHE_DIR / "perspective_fixed"
TASKS = ["pass", "dest", "shot", "interception", "ball_recovery", "pressure", "dribble", "tackle"]
TASK_ID = {t: i for i, t in enumerate(TASKS)}
N_DEST = 96
F_BASE = ["location_x", "location_y", "dist_to_goal", "angle_to_goal", "angle_to_goal_center", "under_pressure"]
F_PASS = ["pass_length", "pass_angle", "pass_end_dist_to_goal", "pass_end_angle_to_goal", "pass_is_progressive",
          "pass_lateral_displacement", "pass_cross", "pass_switch", "pass_through_ball", "pass_cut_back"]
F_SB = ["sb_distance_to_nearest_defender", "sb_num_defenders_on_goal_side", "sb_visible_teammates", "sb_visible_opponents"]
F_FX = ["fx_distance_to_nearest_defender", "fx_num_defenders_on_goal_side", "fx_visible_teammates", "fx_visible_opponents"]
TACKLE_POS = {"Won", "Success", "Success In Play", "Success Out"}
TACKLE_NEG = {"Lost In Play", "Lost Out"}
PRESS_ONBALL = {"Pass", "Carry", "Shot", "Dribble"}
DEV_CAP, TEST_CAP = 100000, 30000


def pressure_labels_5s(df):
    """1 if the pressing team completes a Pass, Carry, Shot or Dribble within 5 s of the pressure start, in the same half. Returns a Series indexed by event_id."""
    t = pd.to_timedelta(df["timestamp"]).dt.total_seconds().values
    key = df["match_id"].values.astype(np.int64) * 10 + df["period"].values.astype(np.int64)
    d = pd.DataFrame({"key": key, "team": df["team_id"].values, "ts": t, "type": df["type_name"].values, "eid": df["event_id"].values})
    nxt = {k: np.sort(g["ts"].values) for k, g in d[d["type"].isin(PRESS_ONBALL)].groupby(["key", "team"])}
    pr = d[d["type"] == "Pressure"]
    y = np.zeros(len(pr), dtype=np.int64)
    for i, (k, tm, ts) in enumerate(zip(pr["key"].values, pr["team"].values, pr["ts"].values)):
        arr = nxt.get((k, tm))
        if arr is not None:
            j = np.searchsorted(arr, ts, side="right")
            if j < len(arr) and arr[j] - ts <= 5.0:
                y[i] = 1
    return pd.Series(y, index=pr["eid"].values)


def build_dataset(smoke=False):
    """Return the dev and test sets of every task (features, labels, tensors, match ids). Subsampling is fixed with numpy seed 42."""
    rng = np.random.RandomState(42)
    cols = ["event_id", "match_id", "period", "timestamp", "team_id", "type_name", "season_dir"] + F_BASE + F_PASS + F_SB + \
           ["pass_end_location_x", "pass_end_location_y", "pass_outcome_name", "shot_outcome_name", "shot_type_name",
            "interception_outcome_name", "ball_recovery_failure"]
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet", columns=cols)
    df = df[df["season_dir"].isin(FULL_SEASONS)].reset_index(drop=True)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    for c in ["pass_is_progressive", "pass_cross", "pass_switch", "pass_through_ball", "pass_cut_back"]:
        df[c] = df[c].fillna(False).astype(int)
    press_y = pressure_labels_5s(df)
    st = pd.read_parquet("data/raw_scan/match_state_2425.parquet")
    normal = set(st.loc[st["state"] == "normal", "match_id"].astype(int))
    is_dev = df["season_dir"].isin(TRAIN_SEASONS).values
    is_test = df["season_dir"].isin(TEST_SEASONS).values & df["match_id"].astype(int).isin(normal).values
    dev_cap, test_cap = (4000, 1500) if smoke else (DEV_CAP, TEST_CAP)

    def pick(d, y, lookup, smaps, X):
        out = {}
        for part, mask, cap in [("dev", d["_dev"].values, dev_cap), ("test", d["_test"].values, test_cap)]:
            idx = np.where(mask)[0]
            if len(idx) > cap:
                idx = np.sort(rng.choice(idx, cap, replace=False))
            rows = np.array([lookup[e] for e in d["event_id"].values[idx]])
            order = np.argsort(rows)
            sm = np.empty((len(rows), 7, 8, 12), dtype=np.float32)
            sm[order] = np.asarray(smaps[rows[order]])
            out[part] = dict(X=X[idx], y=y[idx], sm=sm, match=d["match_id"].values[idx].astype(np.int64),
                             eid=d["event_id"].values[idx].astype(str))
        return out

    data = {}
    df["_dev"], df["_test"] = is_dev, is_test
    df = df[df["_dev"] | df["_test"]]
    zeros_pass = np.zeros(len(F_PASS), dtype=np.float32)

    def feats(d, sb_cols, with_pass):
        base = d[F_BASE].fillna(0).values.astype(np.float32)
        ps = d[F_PASS].fillna(0).values.astype(np.float32) if with_pass else np.tile(zeros_pass, (len(d), 1))
        sb = d[sb_cols].fillna(0).values.astype(np.float32)
        return np.hstack([base, ps, sb])

    for task in TASKS:
        t0 = time.time()
        if task in ("dribble", "tackle"):
            tag = "dribble" if task == "dribble" else "duel"
            idx = pd.read_parquet(FIX_DIR / f"soccermaps_{tag}_idx.parquet")
            idx["_row"] = np.arange(len(idx))
            if task == "dribble":
                idx["y"] = (idx["outcome"] == "Complete").astype(int)
            else:
                idx = idx[(idx["duel_type"] == "Tackle") & idx["outcome"].isin(TACKLE_POS | TACKLE_NEG)].copy()
                idx["y"] = idx["outcome"].isin(TACKLE_POS).astype(int)
            d = idx[["event_id", "_row", "y"] + F_FX].merge(df[["event_id", "match_id", "_dev", "_test"] + F_BASE], on="event_id", how="inner")
            lookup = dict(zip(d["event_id"].values, d["_row"].values))
            smaps = np.load(FIX_DIR / f"soccermaps_{tag}.npy", mmap_mode="r")
            X = feats(d, F_FX, with_pass=False)
            y = d["y"].values.astype(np.int64)
        else:
            src = {"pass": "pass", "dest": "pass", "shot": "shot", "interception": "interception",
                   "ball_recovery": "ball_recovery", "pressure": "pressure"}[task]
            type_name = {"pass": "Pass", "shot": "Shot", "interception": "Interception",
                         "ball_recovery": "Ball Recovery", "pressure": "Pressure"}[src]
            d = df[df["type_name"] == type_name]
            if task == "shot":
                d = d[d["shot_type_name"] != "Penalty"]
            if task == "dest":
                d = d[d["pass_end_location_x"].notna() & d["pass_end_location_y"].notna()]
            sidx = pd.read_parquet(CACHE_DIR / f"action_soccermaps_{src}_idx.parquet")
            lookup = {e: i for i, e in enumerate(sidx["event_id"].values)}
            d = d[d["event_id"].isin(lookup)].reset_index(drop=True)
            smaps = np.load(CACHE_DIR / f"action_soccermaps_{src}.npy", mmap_mode="r")
            X = feats(d, F_SB, with_pass=(task == "pass"))
            if task == "pass":
                y = d["pass_outcome_name"].isna().astype(np.int64).values
            elif task == "dest":
                gx = np.clip((d["pass_end_location_x"].values / 10).astype(int), 0, 11)
                gy = np.clip((d["pass_end_location_y"].values / 10).astype(int), 0, 7)
                y = (gy * 12 + gx).astype(np.int64)
            elif task == "shot":
                y = (d["shot_outcome_name"] == "Goal").astype(np.int64).values
            elif task == "interception":
                y = d["interception_outcome_name"].isin({"Won", "Success In Play"}).astype(np.int64).values
            elif task == "ball_recovery":
                y = (d["ball_recovery_failure"] != True).astype(np.int64).values
            else:
                y = press_y.reindex(d["event_id"].values).fillna(0).values.astype(np.int64)
        data[task] = pick(d, y, lookup, smaps, X)
        print(f"[data] {task:14s} dev {len(data[task]['dev']['y']):7,} test {len(data[task]['test']['y']):6,} "
              f"mean label dev {data[task]['dev']['y'].mean():.3f} test {data[task]['test']['y'].mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return data


def rich_scalars(sm, bx, by):
    """Hand-crafted summary of the 7-channel tensor into 15 scalars (used by --rich_scalars).
    Teammates and opponents, 7 each. Players in the ball cell and in its 3x3 and 5x5 neighbourhoods, players ahead of and
    behind the ball, total visible players, and the cell distance from the ball to the nearest player. Plus the number of
    opponents ahead of the ball within one cell of the line from the ball to the goal centre. sm is (n,7,8,12); bx, by are
    the ball location (yards)."""
    n = len(sm)
    gx = np.clip((bx / 10).astype(int), 0, 11)
    gy = np.clip((by / 10).astype(int), 0, 7)
    yy, xx = np.mgrid[0:8, 0:12]
    dx = xx[None] - gx[:, None, None]
    dy = yy[None] - gy[:, None, None]
    cheb = np.maximum(np.abs(dx), np.abs(dy))
    f = []
    for A in (sm[:, 0], sm[:, 1]):
        f += [(A * (cheb == 0)).sum((1, 2)), (A * (cheb <= 1)).sum((1, 2)), (A * (cheb <= 2)).sum((1, 2)),
              (A * (dx > 0)).sum((1, 2)), (A * (dx < 0)).sum((1, 2)), A.sum((1, 2))]
        f.append(np.where(A > 0, np.sqrt(dx ** 2 + dy ** 2), 99.0).reshape(n, -1).min(1))
    slope = (3.5 - gy[:, None, None]) / np.maximum(11.5 - gx[:, None, None], 0.5)
    lane = (dx > 0) & (np.abs(yy[None] - (gy[:, None, None] + slope * dx)) <= 1.0)
    f.append((sm[:, 1] * lane).sum((1, 2)))
    return np.stack(f, 1).astype(np.float32)


class UnifiedGatingClean(nn.Module):
    def __init__(self, event_dim, n_tasks, fusion="gate", task_embed_dim=16, cnn_channels=7, embed_dim=64, dropout=0.3, hard_gate=False):
        super().__init__()
        self.fusion = fusion
        self.hard_gate = hard_gate
        self.task_embed = nn.Embedding(n_tasks, task_embed_dim)
        self.event_enc = nn.Sequential(nn.Linear(event_dim + task_embed_dim, embed_dim), nn.ELU(), nn.Dropout(dropout),
                                       nn.Linear(embed_dim, embed_dim), nn.Tanh())
        self.spatial_enc = nn.Sequential(
            nn.Conv2d(cnn_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(256, embed_dim), nn.Tanh())
        if fusion == "gate":
            self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        else:
            self.mix = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Tanh())
        self.hidden = nn.Sequential(nn.Linear(embed_dim, 32), nn.ELU(), nn.Dropout(dropout))
        self.out_bin = nn.Linear(32, 1)
        self.out_dest = nn.Linear(32, N_DEST)

    def forward(self, x, s, t):
        h_e = self.event_enc(torch.cat([x, self.task_embed(t)], dim=-1))
        h_s = self.spatial_enc(s)
        if self.fusion == "gate":
            g = self.gate(torch.cat([h_e, h_s], dim=-1))
            if self.hard_gate:
                # Hard gate. Each hidden dimension takes either the spatial or the scalar branch, never a weighted mix, so later
                # layers cannot amplify a very small mixing weight back up. Training samples 0 or 1 with probability g
                # (straight-through estimator for the gradient) and returns g for the penalty term; at test time a dimension is
                # open when g > 0.5 and the returned value is that 0 or 1, so the test-time gate mean is the share of open dimensions.
                if self.training:
                    b = (torch.rand_like(g) < g).float()
                    b = b + g - g.detach()
                    out_g = g
                else:
                    b = (g > 0.5).float()
                    out_g = b
                h = b * h_s + (1 - b) * h_e
                g = out_g
            else:
                h = g * h_s + (1 - g) * h_e
        else:
            g = None
            h = self.mix(torch.cat([h_e, h_s], dim=-1))
        z = self.hidden(h)
        return self.out_bin(z).squeeze(-1), self.out_dest(z), g


class UnifiedGatingFCN(nn.Module):
    """Unified gating model that keeps the spatial resolution. The gate is one 64-d vector per event, as in the pooled version."""

    def __init__(self, event_dim, n_tasks, fusion="gate", task_embed_dim=16, cnn_channels=7, embed_dim=64, dropout=0.3):
        super().__init__()
        self.fusion = fusion
        self.task_embed = nn.Embedding(n_tasks, task_embed_dim)
        self.event_enc = nn.Sequential(nn.Linear(event_dim + task_embed_dim, embed_dim), nn.ELU(), nn.Dropout(dropout),
                                       nn.Linear(embed_dim, embed_dim), nn.Tanh())
        self.spatial_map = nn.Sequential(
            nn.Conv2d(cnn_channels, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.Tanh())
        self.spatial_vec = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(embed_dim * 4, embed_dim), nn.Tanh())
        # Per-cell scalar representation. The event vector is broadcast, two coordinate channels are appended, then a 1x1 conv
        self.event_map = nn.Sequential(nn.Conv2d(embed_dim + 2, embed_dim, 1), nn.Tanh())
        gy, gx = torch.meshgrid(torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 12), indexing="ij")
        self.register_buffer("coords", torch.stack([gx, gy]).unsqueeze(0))
        if fusion == "gate":
            self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        else:
            self.mix = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Tanh())
            self.mix_map = nn.Sequential(nn.Conv2d(embed_dim * 2, embed_dim, 1), nn.Tanh())
        self.hidden = nn.Sequential(nn.Linear(embed_dim, 32), nn.ELU(), nn.Dropout(dropout))
        self.out_bin = nn.Linear(32, 1)
        self.dest_head = nn.Sequential(nn.Conv2d(embed_dim, 32, 1), nn.ELU(), nn.Conv2d(32, 1, 1))

    def forward(self, x, s, t):
        h_e = self.event_enc(torch.cat([x, self.task_embed(t)], dim=-1))
        fmap = self.spatial_map(s)
        h_s = self.spatial_vec(fmap)
        emap = self.event_map(torch.cat([h_e[:, :, None, None].expand(-1, -1, 8, 12), self.coords.expand(len(x), -1, -1, -1)], dim=1))
        if self.fusion == "gate":
            g = self.gate(torch.cat([h_e, h_s], dim=-1))
            h = g * h_s + (1 - g) * h_e
            m = g[:, :, None, None] * fmap + (1 - g)[:, :, None, None] * emap
        else:
            g = None
            h = self.mix(torch.cat([h_e, h_s], dim=-1))
            m = self.mix_map(torch.cat([fmap, emap], dim=1))
        return self.out_bin(self.hidden(h)).squeeze(-1), self.dest_head(m).flatten(1), g


def predict(model, X, S, T, bs=8192):
    model.eval()
    lb, ld, gm = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            b, d, g = model(X[i:i + bs], S[i:i + bs], T[i:i + bs])
            lb.append(b)
            ld.append(torch.log_softmax(d, dim=-1))
            gm.append(g.mean(dim=1) if g is not None else torch.full((len(b),), float("nan")))
    return torch.cat(lb).numpy(), torch.cat(ld).numpy(), torch.cat(gm).numpy()


def task_metrics(task_ids, y, lb, ld):
    out = {}
    for task in TASKS:
        m = task_ids == TASK_ID[task]
        if m.sum() == 0:
            continue
        if task == "dest":
            lp = ld[m]
            yy = y[m].astype(int)
            nll = float(-lp[np.arange(len(yy)), yy].mean())
            top1 = float((lp.argmax(1) == yy).mean())
            top3 = float((np.argsort(-lp, axis=1)[:, :3] == yy[:, None]).any(1).mean())
            out[task] = dict(logloss=nll, top1=top1, top3=top3, skill=1 - nll / math.log(N_DEST))
        else:
            auc = float(roc_auc_score(y[m], lb[m]))
            out[task] = dict(auc=auc, skill=auc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="u1")
    ap.add_argument("--fusion", default="gate", choices=["gate", "concat"])
    ap.add_argument("--encoder", default="pooled", choices=["pooled", "fcn"])
    ap.add_argument("--channels", type=int, default=7, choices=[2, 7])
    ap.add_argument("--shuffle_spatial", default="")
    ap.add_argument("--rich_scalars", action="store_true", help="add 15 hand-crafted summaries of the spatial tensor to the scalar branch")
    ap.add_argument("--hard_gate", action="store_true", help="hard gate, each hidden dimension picks one branch; pooled encoder with gated fusion only")
    ap.add_argument("--gate_penalty", type=float, default=0.0, help="cost of opening the gate, adds coefficient x gate mean to the loss; gated fusion only")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TASG_THREADS", "4")))  # number of threads from TASG_THREADS, default 4
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"{a.tag}_s{a.seed}.json"
    if out_json.exists() and not a.smoke:
        print(f"[skip] {out_json} exists", flush=True)
        return
    lock = OUT_DIR / f"{a.tag}_s{a.seed}.running"  # lock file, so that two queues never run the same configuration at once
    if not a.smoke:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            print(f"[skip] {lock} exists, another process is running this configuration", flush=True)
            return
    t0 = time.time()
    data = build_dataset(a.smoke)
    if a.rich_scalars:
        # Widen the scalar branch. Must run before any shuffling, since the scalars come from the event's own tensor
        for t in TASKS:
            for part in ("dev", "test"):
                d = data[t][part]
                d["X"] = np.hstack([d["X"], rich_scalars(d["sm"], d["X"][:, 0], d["X"][:, 1])])
        print(f"[rich] scalar branch widened to {data[TASKS[0]]['dev']['X'].shape[1]} features  ({time.time() - t0:.0f}s)", flush=True)
    shuffled = [t for t in a.shuffle_spatial.split(",") if t]
    rs = np.random.RandomState(777)
    for t in shuffled:
        for part in ("dev", "test"):
            data[t][part]["sm"] = data[t][part]["sm"][rs.permutation(len(data[t][part]["sm"]))]
        print(f"[intervention] spatial tensors shuffled within task {t}", flush=True)

    dev = {k: np.concatenate([data[t]["dev"][k] for t in TASKS]) for k in ("X", "y", "sm", "match")}
    dev["task"] = np.concatenate([np.full(len(data[t]["dev"]["y"]), TASK_ID[t]) for t in TASKS])
    test = {k: np.concatenate([data[t]["test"][k] for t in TASKS]) for k in ("X", "y", "sm", "match", "eid")}
    test["task"] = np.concatenate([np.full(len(data[t]["test"]["y"]), TASK_ID[t]) for t in TASKS])
    del data

    if a.channels == 2:
        # Keep only the teammate and opponent position channels. Ball location and goal geometry stay in the scalar branch, so the spatial branch carries only the player configuration
        dev["sm"], test["sm"] = np.ascontiguousarray(dev["sm"][:, :2]), np.ascontiguousarray(test["sm"][:, :2])
    matches = np.sort(np.unique(dev["match"]))
    perm = np.random.RandomState(20261002 + a.seed).permutation(len(matches))
    tr = np.isin(dev["match"], matches[perm[:int(0.8 * len(matches))]])
    va = ~tr
    scaler = StandardScaler().fit(dev["X"][tr])
    Xtr = torch.tensor(scaler.transform(dev["X"][tr]), dtype=torch.float32)
    Str = torch.tensor(dev["sm"][tr])
    Ttr = torch.tensor(dev["task"][tr], dtype=torch.long)
    Ytr = torch.tensor(dev["y"][tr], dtype=torch.long)
    Xva = torch.tensor(scaler.transform(dev["X"][va]), dtype=torch.float32)
    Sva = torch.tensor(dev["sm"][va])
    Tva = torch.tensor(dev["task"][va], dtype=torch.long)
    Xte = torch.tensor(scaler.transform(test["X"]), dtype=torch.float32)
    Ste = torch.tensor(test["sm"])
    Tte = torch.tensor(test["task"], dtype=torch.long)
    yva, tva = dev["y"][va], dev["task"][va]
    print(f"[split] train {int(tr.sum()):,} val {int(va.sum()):,} test {len(test['y']):,}  features {Xtr.shape[1]}  ({time.time() - t0:.0f}s)", flush=True)

    t_data = time.time() - t0
    torch.manual_seed(a.seed * 100003 + 7)
    Model = UnifiedGatingClean if a.encoder == "pooled" else UnifiedGatingFCN
    if a.hard_gate:
        assert a.encoder == "pooled" and a.fusion == "gate", "--hard_gate supports only the pooled encoder with gated fusion"
        model = Model(Xtr.shape[1], len(TASKS), fusion=a.fusion, cnn_channels=a.channels, hard_gate=True)
    else:
        model = Model(Xtr.shape[1], len(TASKS), fusion=a.fusion, cnn_channels=a.channels)
    print(f"[model] encoder={a.encoder} fusion={a.fusion} params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    lam = 1.0 / math.log(N_DEST)
    dest_id = TASK_ID["dest"]
    n = len(Xtr)
    best, best_state, bad, hist = -1e9, None, 0, []
    epochs = 1 if a.smoke else a.epochs
    for ep in range(1, epochs + 1):
        model.train()
        perm_i = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, 512):
            b = perm_i[i:i + 512]
            lb, ld, g_b = model(Xtr[b], Str[b], Ttr[b])
            isd = Ttr[b] == dest_id
            loss = torch.zeros(())
            if (~isd).any():
                loss = loss + bce(lb[~isd], Ytr[b][~isd].float())
            if isd.any():
                loss = loss + lam * ce(ld[isd], Ytr[b][isd])
            if a.gate_penalty > 0 and g_b is not None:
                # Gate penalty. When both branches suffice, the gate value is otherwise undetermined; with the penalty the gate opens only where the spatial branch actually helps
                loss = loss + a.gate_penalty * g_b.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        vb, vd, vg = predict(model, Xva, Sva, Tva)
        vm = task_metrics(tva, yva, vb, vd)
        score = float(np.mean([v["skill"] for v in vm.values()]))
        # Per-task validation gate mean at every epoch, to check whether the gate has levelled off when training stops (with a gate penalty it keeps falling during training)
        vgate = {t: round(float(np.nanmean(vg[tva == TASK_ID[t]])), 4) for t in TASKS} if a.fusion == "gate" else None
        hist.append(dict(epoch=ep, train_loss=tot / nb, val_score=score, val_gate=vgate))
        flag = ""
        if score > best:
            best, bad, flag = score, 0, " *"
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        el = time.time() - t0
        left = min(epochs - ep, a.patience - bad)
        print(f"[epoch {ep:2d}/{epochs}] loss {tot / nb:.4f}  val score {score:.4f}{flag}  elapsed {el / 60:.1f} min  "
              f"eta at most {(el - t_data) / ep * (epochs - ep) / 60:.0f} min (stops early after {left} more epochs without improvement)", flush=True)
        if bad >= a.patience:
            break
    model.load_state_dict(best_state)

    tb, td, tg = predict(model, Xte, Ste, Tte)
    vb, vd, _ = predict(model, Xva, Sva, Tva)
    tm, vm = task_metrics(test["task"], test["y"], tb, td), task_metrics(tva, yva, vb, vd)
    # Test-time ablation. Shuffle the test spatial tensors across events of the same task and predict again; the metric drop is how
    # much spatial information the trained model actually uses on that task. Two shuffles. "all" shuffles the whole tensor, "players"
    # only the first two channels (teammate and opponent positions) and keeps the other five, which depend only on the ball location.
    # "players" isolates the contribution of player positions; "all" also includes ball location information.
    rs2 = np.random.RandomState(4242)
    perms = {task: np.where(test["task"] == TASK_ID[task])[0] for task in TASKS}
    perms = {task: (idx, idx[rs2.permutation(len(idx))]) for task, idx in perms.items()}
    Ste_abl = Ste.clone()
    for idx, sh in perms.values():
        Ste_abl[idx] = Ste[sh]
    ab, ad, _ = predict(model, Xte, Ste_abl, Tte)
    am = task_metrics(test["task"], test["y"], ab, ad)
    Ste_abl = Ste.clone()
    for idx, sh in perms.values():
        Ste_abl[idx, :2] = Ste[sh, :2]
    ab, ad, _ = predict(model, Xte, Ste_abl, Tte)
    am_pl = task_metrics(test["task"], test["y"], ab, ad)
    del Ste_abl
    res = dict(tag=a.tag, seed=a.seed, host=socket.gethostname(), fusion=a.fusion, encoder=a.encoder, channels=a.channels, rich_scalars=a.rich_scalars, gate_penalty=a.gate_penalty, hard_gate=a.hard_gate, shuffled=shuffled, best_val_score=best, n_epochs=len(hist),
               history=hist, minutes=round((time.time() - t0) / 60, 1), tasks={})
    print(f"\n{'task':14s} {'test':>22s} {'val':>10s} {'gate_mean':>10s} {'gate_sd':>8s} {'n_test':>7s} {'rel_all':>8s} {'rel_players':>11s}")
    for task in TASKS:
        m = test["task"] == TASK_ID[task]
        r = dict(test=tm[task], val=vm[task], test_spatial_ablated=am[task], n_test=int(m.sum()),
                 spatial_reliance=float(tm[task]["skill"] - am[task]["skill"]),
                 test_players_ablated=am_pl[task], player_reliance=float(tm[task]["skill"] - am_pl[task]["skill"]),
                 gate_mean=float(np.nanmean(tg[m])) if a.fusion == "gate" else None,
                 gate_event_sd=float(np.nanstd(tg[m])) if a.fusion == "gate" else None)
        res["tasks"][task] = r
        ts = f"top1 {tm[task]['top1']:.3f} nll {tm[task]['logloss']:.3f}" if task == "dest" else f"auc {tm[task]['auc']:.4f}"
        print(f"{task:14s} {ts:>22s} {vm[task]['skill']:>10.4f} {r['gate_mean'] if r['gate_mean'] is not None else float('nan'):>10.4f} "
              f"{r['gate_event_sd'] if r['gate_event_sd'] is not None else float('nan'):>8.4f} {int(m.sum()):>7d} {r['spatial_reliance']:>8.4f} {r['player_reliance']:>11.4f}")
    if not a.smoke:
        json.dump(res, open(out_json, "w"), ensure_ascii=False, indent=1)
        torch.save({"state": best_state, "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}, OUT_DIR / f"{a.tag}_s{a.seed}.pt")
        yy = test["y"].astype(int)
        np.savez_compressed(OUT_DIR / f"{a.tag}_s{a.seed}.npz", task=test["task"].astype(np.int8), y=yy.astype(np.int16),
                            logit_bin=tb.astype(np.float32), dest_logp_true=td[np.arange(len(yy)), np.clip(yy, 0, N_DEST - 1)].astype(np.float32),
                            dest_top1=td.argmax(1).astype(np.int16), gate_mean=tg.astype(np.float32),
                            match_id=test["match"], event_id=test["eid"])
        lock.unlink(missing_ok=True)
    print(f"[done] {(time.time() - t0) / 60:.1f} min  {'(smoke, nothing saved)' if a.smoke else out_json}", flush=True)


if __name__ == "__main__":
    main()
