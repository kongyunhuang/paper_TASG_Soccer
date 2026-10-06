#!/usr/bin/env python3
"""
统一多任务门控模型，干净数据版  在清洗后的任务集上重训 U1，并加入传球落点任务
============================
和原来的 train_unified.py 相比有这些不同，模型主体（事件编码器、SoccerMap 编码器、门控、任务嵌入）不变。
  任务集  pass、dest（传球落点所在的 8x12 格，96 类）、shot、interception、ball_recovery、pressure、dribble、tackle
          去掉 Ball Receipt、xG，Duel 只留 Tackle
  数据    dribble 和 tackle 用视角摆正后的张量和重算的 360 标量（data/cache/perspective_fixed/）
          pressure 换成只用球队和时间的标签，逼抢开始后 5 秒内同一半场逼抢方完成 Pass、Carry、Shot、Dribble 之一记为成功
          测试集只用 24/25 里 360 状态正常的 229 场比赛，反转状态的 531 场整场不用
  特征    不用 duration 和 dribble_overrun（动作结束后才知道）。传球的终点、长度、角度等只给 pass 任务，dest 任务这些维度为零
  输出头  二分类任务共用一个 1 维输出，dest 用一个 96 维输出，两者共用同一个融合表示
  损失    二分类用 BCE，dest 用交叉熵乘以 1/ln(96)，使均匀猜测时两类损失同一量级
  消融    训练完后把测试集的空间张量在同任务内打乱再预测一次，记录每个任务指标下降多少（spatial_reliance），
          这是不依赖门控的、对“模型在这个任务上用了多少空间信息”的直接度量。同时保存模型权重
  划分    测试集和各任务的抽样固定（numpy 种子 42），每个种子重新划分训练和验证比赛，并改变初始化和批次顺序
编码器  --encoder pooled 是原来的 SoccerMap 编码器（池化到 2x2 再压成向量），落点任务从融合向量出 96 维
        --encoder fcn 保留 8x12 分辨率。空间分支输出逐格特征图，标量分支把自己的向量铺到每一格并拼上格子坐标，
        同一个门控向量在每一格上混合两者，落点任务逐格出一个 logit。二分类任务的通路不变（特征图池化成向量后门控融合）。
        这样落点的空间信息不会在池化时丢掉，标量分支也能单独给出按位置的落点先验，门控仍然表示每个隐维度取自空间分支的比例
可选的干预  --shuffle_spatial 把指定任务的空间张量在同任务事件之间打乱，空间输入对该任务不再带信息，用来看门控是否随之关小
            --fusion concat 把门控换成拼接后线性映射，作为同一统一模型里的对照

输入  data/cache/L1_events_v3.parquet
      data/cache/action_soccermaps_{pass,shot,interception,ball_recovery,pressure}.npy 和 _idx.parquet
      data/cache/perspective_fixed/soccermaps_{dribble,duel}.npy 和 _idx.parquet
      audit/raw_scan/match_state_2425.parquet
输出  data/cache/u1_clean/{tag}_s{seed}.json   各任务的测试指标、验证指标、门控均值
      data/cache/u1_clean/{tag}_s{seed}.npz    测试集逐事件的预测和门控均值

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/training/train_unified_clean.py --seed 0
  PYTHONPATH=. python -u scripts/training/train_unified_clean.py --seed 0 --shuffle_spatial dest --tag shuf_dest
  PYTHONPATH=. python -u scripts/training/train_unified_clean.py --seed 0 --channels 2 --tag u1_2ch   （空间分支只给球员位置两个通道）
  PYTHONPATH=. python -u scripts/training/train_unified_clean.py --seed 0 --rich_scalars --tag u1rich   （标量分支加 15 个从张量手工汇总的量）
  PYTHONPATH=. python -u scripts/training/train_unified_clean.py --seed 0 --gate_penalty 0.01 --tag u1pen010   （门控开启有代价，损失加 0.01 × 门控均值）
  PYTHONPATH=. python -u scripts/training/train_unified_clean.py --seed 0 --hard_gate --gate_penalty 0.01 --tag u1hard010   （硬门控加代价）
  PYTHONPATH=. python -u scripts/training/train_unified_clean.py --seed 0 --smoke   （每任务几千条跑一轮，只检查能跑通）

Last modified 2026-10-02
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
    """逼抢开始后 5 秒内，同一半场里逼抢方完成一次 Pass、Carry、Shot、Dribble 记 1。返回以 event_id 为索引的 Series。"""
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
    """返回每个任务的开发集和测试集（特征、标签、张量、比赛号）。抽样固定用 numpy 种子 42。"""
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
    st = pd.read_parquet("audit/raw_scan/match_state_2425.parquet")
    normal = set(st.loc[st["state"] == "正常", "match_id"].astype(int))
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
    """从 7 通道张量手工汇总 15 个标量（--rich_scalars 用），照搬第三视角试验七 pilot_rich_scalar_baseline.py 的 summarize。
    队友和对手各 7 个  球所在格、3x3、5x5 邻域内的人数，球前方和后方的人数，可见总人数，最近一人到球的格距离；
    再加球到球门中心连线上下各一格宽的带内、且在球前方的对手数。sm (n,7,8,12)，bx by 是球的位置（码）。"""
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
                # 硬门控  每个隐维度要么取空间分支要么取标量分支，不做加权混合，后面的层就没法把一个很小的混合权重放大回来。
                # 训练时按概率 g 抽 0 或 1（直通估计传梯度），返回 g 供代价项用；测试时 g 过 0.5 的维度开门，返回的是开门的 0 或 1，
                # 所以测试时的门控均值就是开门维度的比例。
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
    """保留空间分辨率的统一门控模型。门控向量和 pooled 版一样是每个事件一个 64 维向量。"""

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
        # 标量分支的逐格表示  事件向量铺开后拼上两个坐标通道，再过 1x1 卷积
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
    ap.add_argument("--rich_scalars", action="store_true", help="标量分支加上从空间张量手工汇总的 15 个量")
    ap.add_argument("--hard_gate", action="store_true", help="硬门控，每个隐维度二选一，只对原编码器的门控融合有效")
    ap.add_argument("--gate_penalty", type=float, default=0.0, help="门控开启的代价系数，损失里加 系数 × 门控均值，只对门控融合有效")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TASG_THREADS", "4")))  # 环境变量 TASG_THREADS 可改线程数，默认 4
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"{a.tag}_s{a.seed}.json"
    if out_json.exists() and not a.smoke:
        print(f"[skip] {out_json} exists", flush=True)
        return
    lock = OUT_DIR / f"{a.tag}_s{a.seed}.running"  # 占位文件，防止两条队列同时跑同一个配置
    if not a.smoke:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            print(f"[skip] {lock} exists, another process is running this configuration", flush=True)
            return
    t0 = time.time()
    data = build_dataset(a.smoke)
    if a.rich_scalars:
        # 加厚标量分支。必须在任何打乱之前算，标量取自事件自己的张量
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
        # 只保留队友和对手两个位置通道。球的位置和球门几何只留在标量分支里，这样空间分支带的只有球员构型
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
        assert a.encoder == "pooled" and a.fusion == "gate", "--hard_gate 只支持原编码器的门控融合"
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
                # 门控代价  两个分支都够用时，没有这一项门控取多少是不确定的；加了它，模型只在空间分支确实带来好处时才开门
                loss = loss + a.gate_penalty * g_b.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        vb, vd, vg = predict(model, Xva, Sva, Tva)
        vm = task_metrics(tva, yva, vb, vd)
        score = float(np.mean([v["skill"] for v in vm.values()]))
        # 每轮记下各任务在验证集上的门控均值，用来看停下时门控是否已经走平（加了门控代价后门控会随训练持续下降）
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
    # 测试时消融  把测试集的空间张量在同任务事件之间打乱再预测一次，指标掉多少就是训练好的模型在该任务上实际用了多少空间信息
    # 两种打乱。all 是整个张量，players 只打乱前两个通道（队友和对手的位置），后五个通道只由球的位置决定，保持不动。
    # players 那一种才是“球员站位信息”的贡献，all 还混着球的位置信息。
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
