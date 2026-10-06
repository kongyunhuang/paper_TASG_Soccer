#!/usr/bin/env python3
"""
球员传球决策偏好的正式检验  偏好超出球队和位置的部分还剩多少，换队后带不带得走
============================
在“传到哪”（8x12 共 96 格）这个任务上，先训练一个不含身份的全局选择模型，再在它的输出上
逐层加位置、球队乘位置、球员三层偏好系数，看球员这一层在留出赛季还有没有预测力。
判定标准在运行前写在 memory_logs/2026-10-02_球员决策偏好_预先写下的判定标准.md，这里照它实现。
  折一  学习 22/23，检验 23/24
  折二  学习 22/23 加 23/24，检验 24/25 的 202 场干净比赛
  M0  全局选择模型，结构直接用 train_unified_clean.UnifiedGatingFCN，只训落点任务，只用该折学习赛季的传球
  M1  加位置偏好   M2  再加球队乘位置偏好   M3  再加球员偏好
  偏好项是 5 个系数乘以每个候选格的 5 个属性（向前、横向、距离、落点 3x3 内对手数、是否长传）
  主要的量 D 等于 M3 减 M2 的每次传球对数损失，按球员重抽样给区间，全体和换队球员分开判

输入  data/cache/L1_events_v3.parquet
      data/cache/action_soccermaps_pass.npy 和 action_soccermaps_pass_idx.parquet（只读，内存映射）
      audit/raw_scan/match_state_2425.parquet（定出 24/25 的 202 场干净比赛）
输出  data/cache/player_pref/fold{折}_s{种子}_players.parquet   每名球员各模型的对数损失
      data/cache/player_pref/fold{折}_s{种子}.json               该折该种子的汇总和选中的正则强度
      data/cache/player_pref/global_fold{折}_s{种子}.pt           全局模型权重
      audit/player_decision_preference_summary.md              --summarize 生成，含判定

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/eval/player_decision_preference.py --folds 1,2 --seeds 0,1,2
  PYTHONPATH=. python -u scripts/eval/player_decision_preference.py --summarize
  PYTHONPATH=. python -u scripts/eval/player_decision_preference.py --smoke      （小样本跑通全流程，结果不看）
  线程数用环境变量 TASG_THREADS 控制，默认 4。可断点续跑，已有结果的折和种子自动跳过。

Last modified 2026-10-02
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from scripts.training.train_all import CACHE_DIR
from scripts.training.train_unified_clean import F_BASE, F_PASS, F_SB, N_DEST, TASK_ID, TASKS, UnifiedGatingFCN

OUT_DIR = CACHE_DIR / "player_pref"
SUMMARY_MD = Path("audit/player_decision_preference_summary.md")
S2223 = ["2_235_2022_23", "11_235_2022_23"]
S2324 = ["2_281_2023_24", "11_281_2023_24"]
S2425 = ["2_317_2024_25", "11_317_2024_25"]
FOLDS = {1: dict(fit=S2223, test=S2324, min_fit=300, min_test=300),
         2: dict(fit=S2223 + S2324, test=S2425, min_fit=300, min_test=100)}
MIN_COEF_PASSES = 100      # 学习赛季传球不少于这么多的球员才有自己的系数
MIN_TP_PASSES = 50         # 球队乘位置组合学习赛季不少于这么多次传球才有自己的系数
LAM_GRID = [30.0, 100.0, 300.0, 1000.0, 3000.0]
PHI_NAMES = ["向前", "横向", "距离", "落点对手数", "长传"]
CX = (torch.arange(12).float() * 10 + 5).repeat(8)            # 每格中心 x，长度 96
CY = (torch.arange(8).float() * 10 + 5).repeat_interleave(12)
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def clean_matches_2425():
    """24/25 的干净比赛  360 状态正常且射门带帧率不低于 70%，和 summarize_u1_clean.clean_matches 同一口径（202 场）"""
    st = pd.read_parquet("audit/raw_scan/match_state_2425.parquet")
    return set(st.loc[(st["state"] == "正常") & (st["shot_frame_pct"] >= 70), "match_id"].astype(int))


def load_passes(smoke):
    cols = ["event_id", "match_id", "season_dir", "type_name", "team_id", "player_id", "position_name",
            "pass_end_location_x", "pass_end_location_y"] + F_BASE + F_SB
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet", columns=cols, filters=[("type_name", "==", "Pass")])
    idx = pd.read_parquet(CACHE_DIR / "action_soccermaps_pass_idx.parquet")
    idx["row"] = np.arange(len(idx))
    df = df.merge(idx, on="event_id", how="inner")
    df = df.dropna(subset=["location_x", "location_y", "pass_end_location_x", "pass_end_location_y", "player_id"])
    df = df[df["season_dir"].isin(S2223 + S2324 + S2425)]
    clean = clean_matches_2425()
    df = df[~df["season_dir"].isin(S2425) | df["match_id"].astype(int).isin(clean)]
    if smoke:
        df = df[df["match_id"].astype(int) % 5 == 0]
    df = df.sort_values("row").reset_index(drop=True)
    df["player_id"] = df["player_id"].astype(int)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    df["position_name"] = df["position_name"].fillna("NA")
    gx = np.clip((df["pass_end_location_x"].values / 10).astype(int), 0, 11)
    gy = np.clip((df["pass_end_location_y"].values / 10).astype(int), 0, 7)
    df["y"] = gy * 12 + gx
    base = df[F_BASE].fillna(0).values.astype(np.float32)
    sb = df[F_SB].fillna(0).values.astype(np.float32)
    X = np.hstack([base, np.zeros((len(df), len(F_PASS)), dtype=np.float32), sb])   # 和统一模型落点任务的特征布局相同
    log(f"带帧传球 {len(df):,} 个，其中 24/25 干净比赛 {df['season_dir'].isin(S2425).sum():,} 个（{len(clean)} 场）")
    return df, X


def phi(sm2, x0, y0):
    """每个候选格的 5 个属性，形状 (n, 96, 5)。sm2 是队友和对手两个占位通道。
    坐标约定出自厂商说明书（API 360 Frames v2.0.0 第 3 页和 Events v8.0.0），单位是码，球场 120 乘 80，
    事件所属球队从 x=0 攻向 x=120，y 轴向下。所以 x 增大就是向前；横向只取 y 差的绝对值，和 y 轴朝向无关。
    格子划分和 TASG 落点任务相同，每格 10 码见方，共 8 行 12 列。"""
    oppn = F.conv2d(sm2[:, 1:2], torch.ones(1, 1, 3, 3), padding=1).flatten(1)
    fwd = (CX[None] - x0[:, None]) / 10
    lat = (CY[None] - y0[:, None]).abs() / 10
    dist = torch.sqrt((CX[None] - x0[:, None]) ** 2 + (CY[None] - y0[:, None]) ** 2) / 10
    return torch.stack([fwd, lat, dist, oppn, (dist > 3).float()], dim=2)


def train_global(df, X, S, fit_mask, seed, a, tag):
    """只用学习赛季的传球训练全局选择模型，学习赛季内部按比赛留 15% 做早停。返回模型和特征标准化器。"""
    rng = np.random.RandomState(20261002 + seed)
    matches = np.sort(df.loc[fit_mask, "match_id"].unique())
    perm = rng.permutation(len(matches))
    va_m = set(matches[perm[:max(1, int(0.15 * len(matches)))]])
    is_va = fit_mask & df["match_id"].isin(va_m).values
    is_tr = fit_mask & ~is_va
    tr_i, va_i = np.where(is_tr)[0], np.where(is_va)[0]
    if len(tr_i) > a.train_cap:
        tr_i = np.sort(rng.choice(tr_i, a.train_cap, replace=False))
    if len(va_i) > 60000:
        va_i = np.sort(rng.choice(va_i, 60000, replace=False))
    sc = StandardScaler().fit(X[tr_i])
    rows = df["row"].values
    Str = torch.tensor(np.ascontiguousarray(S[rows[tr_i]]))
    Sva = torch.tensor(np.ascontiguousarray(S[rows[va_i]]))
    Xtr, Xva = torch.tensor(sc.transform(X[tr_i]), dtype=torch.float32), torch.tensor(sc.transform(X[va_i]), dtype=torch.float32)
    ytr, yva = torch.tensor(df["y"].values[tr_i], dtype=torch.long), torch.tensor(df["y"].values[va_i], dtype=torch.long)
    tid = TASK_ID["dest"]
    torch.manual_seed(seed * 100003 + 7)
    model = UnifiedGatingFCN(X.shape[1], len(TASKS))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, bad = 1e9, None, 0
    log(f"{tag} 全局模型 训练 {len(tr_i):,} 验证 {len(va_i):,}")
    for ep in range(a.epochs):
        model.train()
        p = torch.randperm(len(tr_i))
        tot = 0.0
        for i in range(0, len(p), 512):
            b = p[i:i + 512]
            _, ld, _ = model(Xtr[b], Str[b], torch.full((len(b),), tid, dtype=torch.long))
            loss = F.cross_entropy(ld, ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            v = torch.cat([model(Xva[i:i + 8192], Sva[i:i + 8192], torch.full((len(Xva[i:i + 8192]),), tid, dtype=torch.long))[1]
                           for i in range(0, len(va_i), 8192)])
            vl = F.cross_entropy(v, yva).item()
            top1 = (v.argmax(1) == yva).float().mean().item()
        flag = ""
        if vl < best - 1e-4:
            best, bad, flag = vl, 0, " *"
            state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        log(f"{tag} 全局模型 第 {ep + 1}/{a.epochs} 轮 训练损失 {tot / len(tr_i):.4f} 验证损失 {vl:.4f} top1 {top1:.4f}{flag}")
        if bad >= 2:
            break
    model.load_state_dict(state)
    model.eval()
    return model, sc, best


def infer(model, sc, df, X, S, use_mask, tag):
    """对用到的全部传球算全局模型的 log 概率和候选格属性"""
    idx = np.where(use_mask)[0]
    rows = df["row"].values[idx]
    x0 = torch.tensor(df["location_x"].values[idx], dtype=torch.float32)
    y0 = torch.tensor(df["location_y"].values[idx], dtype=torch.float32)
    logp = torch.empty(len(idx), N_DEST)
    feats = torch.empty(len(idx), N_DEST, 5, dtype=torch.float16)
    tid = TASK_ID["dest"]
    with torch.no_grad():
        for i in range(0, len(idx), 20000):
            sm = torch.tensor(np.ascontiguousarray(S[rows[i:i + 20000]]))
            xb = torch.tensor(sc.transform(X[idx[i:i + 20000]]), dtype=torch.float32)
            _, ld, _ = model(xb, sm, torch.full((len(xb),), tid, dtype=torch.long))
            logp[i:i + 20000] = torch.log_softmax(ld, 1)
            feats[i:i + 20000] = phi(sm[:, :2], x0[i:i + 20000], y0[i:i + 20000]).half()
            if (i // 20000) % 15 == 0:
                log(f"{tag} 推断 {min(i + 20000, len(idx)):,}/{len(idx):,}")
    return idx, logp, feats


def fit_levels(logp, feats, y, gids, rows, sizes, lams, epochs, seed):
    """冻结全局模型，学若干层偏好系数（每层每组 5 个），各层相加。第 0 号组的系数固定为零。
    rows 是参与拟合的行号（按行号取数，不整块拷贝）。gids 是和全表等长的分组编号。
    惩罚是每层 lam 乘该层系数平方和，按总和口径（除以样本数后加到平均损失上）。"""
    torch.manual_seed(seed)
    embs = [nn.Embedding(s, 5, padding_idx=0) for s in sizes]
    for e in embs:
        nn.init.zeros_(e.weight)
    opt = torch.optim.Adam([e.weight for e in embs], lr=0.01)
    n = len(rows)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 8192):
            b = rows[perm[i:i + 8192]]
            th = sum(e(g[b]) for e, g in zip(embs, gids))
            lg = logp[b] + (feats[b].float() * th[:, None, :]).sum(2)
            loss = F.cross_entropy(lg, y[b]) + sum(l * (e.weight ** 2).sum() for l, e in zip(lams, embs)) / n
            opt.zero_grad()
            loss.backward()
            opt.step()
    return [e.weight.detach().clone() for e in embs]


def nll(logp, feats, y, W, gids, rows):
    """rows 这些行上每次传球的对数损失。W 为空表示只用全局模型。"""
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 50000):
            b = rows[i:i + 50000]
            lg = logp[b]
            if W:
                th = sum(w[g[b]] for w, g in zip(W, gids))
                lg = lg + (feats[b].float() * th[:, None, :]).sum(2)
            out.append(F.cross_entropy(lg, y[b], reduction="none"))
    return torch.cat(out)


def modal(s):
    return s.mode().iloc[0]


def boot_ci(delta, weight, cluster, n_boot=2000, seed=0):
    """按 cluster（球员）重抽样的加权平均及 95% 分位区间。delta weight cluster 是等长数组，一行是一名球员在一折里的数。"""
    d = pd.DataFrame({"d": delta, "w": weight, "c": cluster})
    g = d.assign(dw=d["d"] * d["w"]).groupby("c")[["dw", "w"]].sum()
    dw, w = g["dw"].values, g["w"].values
    if len(g) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(g), len(g))
        bs.append(dw[i].sum() / w[i].sum())
    return float(dw.sum() / w.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run_fold(fold, seed, df, X, S, a):
    cfg = FOLDS[fold]
    tag = f"折{fold} 种子{seed}"
    out_json = OUT_DIR / f"fold{fold}_s{seed}.json"
    if out_json.exists():
        log(f"{tag} 已有结果，跳过")
        return
    min_fit, min_test = (40, 20) if a.smoke else (cfg["min_fit"], cfg["min_test"])
    min_coef, min_tp = (15, 10) if a.smoke else (MIN_COEF_PASSES, MIN_TP_PASSES)
    is_fit = df["season_dir"].isin(cfg["fit"]).values
    is_test = df["season_dir"].isin(cfg["test"]).values
    model, sc, gval = train_global(df, X, S, is_fit, seed, a, tag)
    torch.save(model.state_dict(), OUT_DIR / f"global_fold{fold}_s{seed}.pt")
    idx, logp, feats = infer(model, sc, df, X, S, is_fit | is_test, tag)
    d = df.iloc[idx].reset_index(drop=True)
    y = torch.tensor(d["y"].values, dtype=torch.long)
    fit = torch.tensor(d["season_dir"].isin(cfg["fit"]).values)
    test = ~fit

    # 分组编号，0 号留给“没有自己的系数”
    dfit = d[fit.numpy()]
    pos_names = sorted(dfit["position_name"].unique())
    pos_map = {p: i + 1 for i, p in enumerate(pos_names)}
    d["pos_id"] = d["position_name"].map(pos_map).fillna(0).astype(int)
    d["tp_key"] = d["team_id"].astype(int).astype(str) + "|" + d["position_name"]
    tp_n = d.loc[fit.numpy(), "tp_key"].value_counts()
    tp_map = {k: i + 1 for i, k in enumerate(sorted(tp_n[tp_n >= min_tp].index))}
    d["tp_id"] = d["tp_key"].map(tp_map).fillna(0).astype(int)
    n_fit = dfit.groupby("player_id").size()
    n_test = d[test.numpy()].groupby("player_id").size()
    coef_players = sorted(n_fit[n_fit >= min_coef].index)
    pl_map = {p: i + 1 for i, p in enumerate(coef_players)}
    d["pl_id"] = d["player_id"].map(pl_map).fillna(0).astype(int)
    pos, tp, pl = (torch.tensor(d[c].values) for c in ("pos_id", "tp_id", "pl_id"))
    sizes = [len(pos_map) + 1, len(tp_map) + 1, len(pl_map) + 1]
    eligible = sorted(set(n_fit[n_fit >= min_fit].index) & set(n_test[n_test >= min_test].index))
    log(f"{tag} 位置 {len(pos_map)} 组，球队乘位置 {len(tp_map)} 组，有系数的球员 {len(pl_map)} 人，进入检验 {len(eligible)} 人")

    # 逐层选正则强度，只在学习赛季内部切验证
    fm = np.sort(dfit["match_id"].unique())
    perm = np.random.RandomState(20261002 + 31 * seed).permutation(len(fm))
    inner_val = set(fm[perm[:int(0.3 * len(fm))]])
    iv_mask = fit & torch.tensor(d["match_id"].isin(inner_val).values)
    r_fit, r_test = torch.where(fit)[0], torch.where(test)[0]
    r_iv, r_it = torch.where(iv_mask)[0], torch.where(fit & ~iv_mask)[0]
    levels = [("位置", pos), ("球队乘位置", tp), ("球员", pl)]
    chosen, path = [], {}
    ep = 2 if a.smoke else 12
    for k, (name, _) in enumerate(levels):
        best_l, best_v = None, 1e9
        for lam in (LAM_GRID[1:3] if a.smoke else LAM_GRID):
            g_all = [g for _, g in levels[:k + 1]]
            W = fit_levels(logp, feats, y, g_all, r_it, sizes[:k + 1], chosen + [lam], ep, seed)
            v = float(nll(logp, feats, y, W, g_all, r_iv).mean())
            path[f"{name}_lam{lam:g}"] = v
            log(f"{tag} 选正则 {name} λ={lam:g} 内部验证对数损失 {v:.5f}")
            if v < best_v:
                best_l, best_v = lam, v
        chosen.append(best_l)
    log(f"{tag} 选中的正则强度 位置 {chosen[0]:g} 球队乘位置 {chosen[1]:g} 球员 {chosen[2]:g}")

    # 在全部学习赛季上拟合 M1 M2 M3，在检验赛季上算每次传球的对数损失
    res_nll = {"M0": nll(logp, feats, y, None, None, r_test).numpy()}
    W3 = None
    for k, mname in enumerate(["M1", "M2", "M3"]):
        g_all = [g for _, g in levels[:k + 1]]
        W = fit_levels(logp, feats, y, g_all, r_fit, sizes[:k + 1], chosen[:k + 1], ep, seed)
        res_nll[mname] = nll(logp, feats, y, W, g_all, r_test).numpy()
        if mname == "M3":
            W3 = W
        log(f"{tag} {mname} 检验赛季全部传球的对数损失 {res_nll[mname].mean():.5f}（M0 {res_nll['M0'].mean():.5f}）")
    dt = d[test.numpy()].copy()
    for m, v in res_nll.items():
        dt[m] = v
    dt = dt[dt["player_id"].isin(eligible)]
    by = dt.groupby("player_id").agg(n_test=("M0", "size"), M0=("M0", "mean"), M1=("M1", "mean"), M2=("M2", "mean"), M3=("M3", "mean"))
    by["n_fit"] = n_fit.reindex(by.index).values
    last_fit = cfg["fit"][-2:]                       # 学习期最后一个赛季（两个联赛）
    team_last = dfit[dfit["season_dir"].isin(last_fit)].groupby("player_id")["team_id"].agg(modal)
    team_any = dfit.groupby("player_id")["team_id"].agg(modal)
    team_fit = team_last.reindex(by.index).fillna(team_any.reindex(by.index))
    team_test = d[test.numpy()].groupby("player_id")["team_id"].agg(modal).reindex(by.index)
    by["mover"] = (team_fit.values != team_test.values)
    by["fold"], by["seed"] = fold, seed

    # 不参与判定的量  球员系数的跨期稳定性。在检验赛季上独立再拟合一次三层模型
    stab = {}
    try:
        epl = torch.tensor(d["player_id"].map({p: i + 1 for i, p in enumerate(eligible)}).fillna(0).astype(int).values)
        sz = [sizes[0], sizes[1], len(eligible) + 1]
        Wa = fit_levels(logp, feats, y, [pos, tp, epl], r_fit, sz, chosen, ep, seed)[2][1:].numpy()
        Wb = fit_levels(logp, feats, y, [pos, tp, epl], r_test, sz, chosen, ep, seed)[2][1:].numpy()
        mv = by["mover"].reindex(eligible).values.astype(bool)
        for j, nm in enumerate(PHI_NAMES):
            stab[nm] = dict(全部=float(np.corrcoef(Wa[:, j], Wb[:, j])[0, 1]),
                            留队=float(np.corrcoef(Wa[~mv, j], Wb[~mv, j])[0, 1]) if (~mv).sum() > 5 else None,
                            换队=float(np.corrcoef(Wa[mv, j], Wb[mv, j])[0, 1]) if mv.sum() > 5 else None)
        za, zb = (Wa - Wa.mean(0)) / (Wa.std(0) + 1e-9), (Wb - Wb.mean(0)) / (Wb.std(0) + 1e-9)
        D = ((za[:, None, :] - zb[None, :, :]) ** 2).sum(2)
        rank = (D < np.diag(D)[:, None]).sum(1)
        stab["再认前十命中率"] = dict(全部=float((rank < 10).mean()), 换队=float((rank[mv] < 10).mean()) if mv.sum() else None,
                               随机=10 / len(eligible))
    except Exception as e:                            # 稳定性是附带的量，出错不影响主结果
        stab["error"] = repr(e)

    with torch.no_grad():
        top1 = float((logp[r_test].argmax(1) == y[r_test]).float().mean())
    summ = dict(fold=fold, seed=seed, smoke=a.smoke, n_eligible=len(by), n_movers=int(by["mover"].sum()),
                lam=dict(位置=chosen[0], 球队乘位置=chosen[1], 球员=chosen[2]), lam_path=path, global_val_loss=gval,
                global_test_logloss=float(res_nll["M0"].mean()), global_test_top1=top1, stability=stab,
                sizes=dict(位置=len(pos_map), 球队乘位置=len(tp_map), 球员=len(pl_map)),
                coef_sd=dict(位置=float(W3[0][1:].std()), 球队乘位置=float(W3[1][1:].std()), 球员=float(W3[2][1:].std())))
    for grp, g in [("全部", by), ("留队", by[~by["mover"]]), ("换队", by[by["mover"]])]:
        summ[grp] = {}
        for nm, a_, b_ in [("M1减M0", "M1", "M0"), ("M2减M1", "M2", "M1"), ("M3减M2", "M3", "M2"), ("M3减M0", "M3", "M0")]:
            m, lo, hi = boot_ci((g[a_] - g[b_]).values, g["n_test"].values, g.index.values)
            summ[grp][nm] = dict(mean=m, ci=[lo, hi])
        summ[grp]["n"] = int(len(g))
        log(f"{tag} {grp}（{len(g)} 人） M3减M2 {summ[grp]['M3减M2']['mean']:+.5f} "
            f"[{summ[grp]['M3减M2']['ci'][0]:+.5f}, {summ[grp]['M3减M2']['ci'][1]:+.5f}]  M3减M0 {summ[grp]['M3减M0']['mean']:+.5f}")
    by.reset_index().to_parquet(OUT_DIR / f"fold{fold}_s{seed}_players.parquet")
    json.dump(summ, open(out_json, "w"), ensure_ascii=False, indent=1)
    log(f"{tag} 结果写入 {out_json}")


def verdict(pooled, f1, f2, seed_means):
    """按预先写下的标准判。pooled 是 (均值, 下沿, 上沿)，f1 f2 是两折各自的点估计，seed_means 是各种子合并点估计"""
    m, lo, hi = pooled
    if np.isnan(m):
        return "无数据"
    if hi < 0 and f1 < 0 and f2 < 0:
        return "成立" if all(s < 0 for s in seed_means) else "不确定（种子平均后成立，但有种子的点估计不为负）"
    if m >= 0 or lo > -0.001:
        return "不成立"
    return "不确定"


def summarize():
    files = sorted(OUT_DIR.glob("fold*_s*_players.parquet"))
    if not files:
        print("没有结果文件")
        return
    allp = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    js = [json.load(open(f)) for f in sorted(OUT_DIR.glob("fold*_s*.json"))]
    smoke = any(j.get("smoke") for j in js)
    allp["d32"] = allp["M3"] - allp["M2"]
    for c in ("d10", "d21", "d30"):
        a_, b_ = {"d10": ("M1", "M0"), "d21": ("M2", "M1"), "d30": ("M3", "M0")}[c]
        allp[c] = allp[a_] - allp[b_]
    # 先在种子之间取平均，一行是一名球员在一折里
    avg = allp.groupby(["fold", "player_id"]).agg(d32=("d32", "mean"), d10=("d10", "mean"), d21=("d21", "mean"), d30=("d30", "mean"),
                                                  M0=("M0", "mean"), n_test=("n_test", "first"), mover=("mover", "first"),
                                                  n_seeds=("seed", "nunique")).reset_index()
    lines = ["# 球员传球决策偏好，正式检验的汇总", "",
             "由 `scripts/eval/player_decision_preference.py --summarize` 生成。判定标准见 "
             "`memory_logs/2026-10-02_球员决策偏好_预先写下的判定标准.md`，这里照它判，不改阈值。"
             + ("**这是 --smoke 小样本通路测试的输出，结果不能看。**" if smoke else ""), "",
             f"已有的折和种子  {sorted(set((int(j['fold']), int(j['seed'])) for j in js))}", ""]
    rows = []
    for grp, sel in [("全体球员", avg["mover"].notna()), ("留队球员", ~avg["mover"].astype(bool)), ("换队球员", avg["mover"].astype(bool))]:
        g = avg[sel.values]
        pooled = boot_ci(g["d32"].values, g["n_test"].values, g["player_id"].values)
        fold_pts = {}
        for f in (1, 2):
            gf = g[g["fold"] == f]
            fold_pts[f] = boot_ci(gf["d32"].values, gf["n_test"].values, gf["player_id"].values) if len(gf) else (float("nan"),) * 3
        seed_means = []
        for s in sorted(allp["seed"].unique()):
            gs = allp[allp["seed"] == s]
            if grp == "留队球员":
                gs = gs[~gs["mover"].astype(bool)]
            elif grp == "换队球员":
                gs = gs[gs["mover"].astype(bool)]
            if len(gs):
                seed_means.append(float(np.average(gs["d32"], weights=gs["n_test"])))
        v = verdict(pooled, fold_pts[1][0], fold_pts[2][0], seed_means) if grp != "留队球员" else "不单独判"
        base = float(np.average(g["M0"], weights=g["n_test"])) if len(g) else float("nan")
        rows.append({"组": grp, "球员数（两折合计行数）": len(g), "D 合并": f"{pooled[0]:+.5f}", "95% 区间": f"[{pooled[1]:+.5f}, {pooled[2]:+.5f}]",
                     "折一": f"{fold_pts[1][0]:+.5f} [{fold_pts[1][1]:+.5f}, {fold_pts[1][2]:+.5f}]",
                     "折二": f"{fold_pts[2][0]:+.5f} [{fold_pts[2][1]:+.5f}, {fold_pts[2][2]:+.5f}]",
                     "各种子合并点估计": " ".join(f"{s:+.5f}" for s in seed_means), "D 占全局模型对数损失": f"{pooled[0] / base * 100:+.2f}%", "判定": v})
    lines += ["## 主要的量 D（M3 减 M2 的每次传球对数损失，负数表示球员偏好在球队和位置之外还有预测力）", "",
              pd.DataFrame(rows).to_markdown(index=False), ""]
    mv, sy = avg[avg["mover"].astype(bool)], avg[~avg["mover"].astype(bool)]
    if len(mv) and len(sy):
        r = np.average(mv["d32"], weights=mv["n_test"]) / np.average(sy["d32"], weights=sy["n_test"])
        lines += [f"换队球员的 D 是留队球员的 {r:.2f} 倍。", ""]
    rows = []
    for grp, g in [("全体球员", avg), ("留队球员", sy), ("换队球员", mv)]:
        r = {"组": grp}
        for c, nm in [("d10", "M1减M0（位置）"), ("d21", "M2减M1（球队乘位置）"), ("d32", "M3减M2（球员）"), ("d30", "M3减M0（合计）")]:
            m, lo, hi = boot_ci(g[c].values, g["n_test"].values, g["player_id"].values)
            r[nm] = f"{m:+.5f} [{lo:+.5f}, {hi:+.5f}]"
        rows.append(r)
    lines += ["## 各层的贡献（不参与判定）", "", pd.DataFrame(rows).to_markdown(index=False), ""]
    rows = []
    for j in js:
        r = {"折": j["fold"], "种子": j["seed"], "进入检验": j["n_eligible"], "换队": j["n_movers"], "全局模型检验 top1": round(j["global_test_top1"], 4),
             "全局模型检验对数损失": round(j["global_test_logloss"], 4), "λ 位置": j["lam"]["位置"], "λ 球队乘位置": j["lam"]["球队乘位置"], "λ 球员": j["lam"]["球员"]}
        st = j.get("stability", {})
        for nm in PHI_NAMES:
            if nm in st:
                r[f"系数相关 {nm}（全部/换队）"] = f"{st[nm]['全部']:.2f}/{st[nm]['换队'] if st[nm]['换队'] is None else round(st[nm]['换队'], 2)}"
        if "再认前十命中率" in st:
            r["再认前十（全部/换队/随机）"] = "/".join(str(None if v is None else round(v, 3)) for v in st["再认前十命中率"].values())
        rows.append(r)
    lines += ["## 每折每个种子的设置和附带的量（不参与判定）", "", pd.DataFrame(rows).to_markdown(index=False), ""]
    out = SUMMARY_MD if not smoke else OUT_DIR / "smoke_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n汇总写入 {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="1,2")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--train_cap", type=int, default=400000, help="全局模型训练用的传球数上限，从学习赛季里随机抽")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    global OUT_DIR
    if a.smoke:
        OUT_DIR = CACHE_DIR / "player_pref_smoke"
        a.train_cap, a.epochs = 20000, 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        summarize()
        return
    torch.set_num_threads(int(os.environ.get("TASG_THREADS", "4")))
    df, X = load_passes(a.smoke)
    S = np.load(CACHE_DIR / "action_soccermaps_pass.npy", mmap_mode="r")
    todo = [(f, s) for f in [int(x) for x in a.folds.split(",")] for s in ([0] if a.smoke else [int(x) for x in a.seeds.split(",")])]
    for k, (f, s) in enumerate(todo):
        t1 = time.time()
        run_fold(f, s, df, X, S, a)
        el = (time.time() - T0) / 60
        log(f"进度 {k + 1}/{len(todo)}，本项用时 {(time.time() - t1) / 60:.1f} 分钟，已用 {el:.1f} 分钟，预计还要 {el / (k + 1) * (len(todo) - k - 1):.0f} 分钟")
    if a.smoke:
        summarize()


if __name__ == "__main__":
    main()
