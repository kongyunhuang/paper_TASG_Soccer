#!/usr/bin/env python3
"""
留出数据确认，标记直接检验  用张量缓存里的队友、对手人数对厂商 visible_player_counts，逐事件查 teammate 标记有没有反
============================
独立审阅的建议。五个任务的张量缓存是 4 月从本地定格帧建的，第 0 通道按队友每人加 1、第 1 通道按对手每人加 1，
通道求和就是帧里标为队友、对手的点数。事件表里 sb_visible_teammates、sb_visible_opponents 是厂商按球队给的可见人数
（核查报告判标记反转的主力规则就是拿帧里的队友点数对它）。逐事件比
  一致    通道 0 之和 等于 sb_visible_teammates 且 通道 1 之和 等于 sb_visible_opponents
  反转    通道 0 之和 等于 sb_visible_opponents 且 通道 1 之和 等于 sb_visible_teammates 且两者不等
  平局    sb 两个数相等（判不了）
  其他    都不是（比如厂商人数缺失）
按任务、赛季组（22/23、23/24、24/25 干净 202、24/25 其余 558、25/26）汇总，25/26 再按场列出反转比例。
这是对五个任务在 146 场上“标记是否反转”的直接检验；坐标镜像在这五类事件上所有状态都是 0%（证据表 T4a、T4b），不另查。
不涉及任何模型。

输入  data/cache/L1_events_v3.parquet，data/cache/action_soccermaps_{task}.npy 和 _idx.parquet，audit/raw_scan/match_state_2425.parquet
输出  audit/holdout_2526_flag_check.md，data/cache/holdout_2526/flag_check_by_match_2526.parquet

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/eval/holdout_2526_flag_check.py

Last modified 2026-10-03
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.eval.summarize_u1_clean import clean_matches

CACHE = Path("data/cache")
OUT_MD = Path("audit/holdout_2526_flag_check.md")
OUT_PQ = CACHE / "holdout_2526" / "flag_check_by_match_2526.parquet"
TASKS = {"pass": "Pass", "shot": "Shot", "interception": "Interception", "ball_recovery": "Ball Recovery", "pressure": "Pressure"}
ORDER = ["22/23", "23/24", "24/25 干净 202", "24/25 其余 558", "25/26"]
CHUNK = 100000


def grp_of(df, clean):
    s = df["season_dir"].astype(str)
    g = np.where(s.str.contains("2022_23"), "22/23", np.where(s.str.contains("2023_24"), "23/24", np.where(s.str.contains("2025_26"), "25/26", "")))
    g = pd.Series(g, index=df.index)
    m = s.str.contains("2024_25")
    g[m & df["match_id"].isin(clean)] = "24/25 干净 202"
    g[m & ~df["match_id"].isin(clean)] = "24/25 其余 558"
    return g


def main():
    t0 = time.time()
    clean = clean_matches()
    df = pd.read_parquet(CACHE / "L1_events_v3.parquet", columns=["event_id", "match_id", "season_dir", "type_name", "sb_visible_teammates", "sb_visible_opponents"])
    df["match_id"] = df["match_id"].astype(int)
    df["grp"] = grp_of(df, clean)
    rows, per_match = [], []
    for task, tn in TASKS.items():
        sidx = pd.read_parquet(CACHE / f"action_soccermaps_{task}_idx.parquet")
        pos = pd.Series(np.arange(len(sidx)), index=sidx["event_id"].values)
        d = df[(df["type_name"] == tn) & df["event_id"].isin(pos.index)].copy()
        d["_row"] = pos.loc[d["event_id"].values].values
        d = d.sort_values("_row").reset_index(drop=True)
        smaps = np.load(CACHE / f"action_soccermaps_{task}.npy", mmap_mode="r")
        n_t = np.empty(len(d), dtype=np.float32)
        n_o = np.empty(len(d), dtype=np.float32)
        r = d["_row"].values
        for i in range(0, len(d), CHUNK):
            blk = np.asarray(smaps[r[i:i + CHUNK], :2])
            n_t[i:i + CHUNK] = blk[:, 0].sum((1, 2))
            n_o[i:i + CHUNK] = blk[:, 1].sum((1, 2))
        vt, vo = d["sb_visible_teammates"].values, d["sb_visible_opponents"].values
        have = ~np.isnan(vt) & ~np.isnan(vo)
        tie = have & (vt == vo)
        agree = have & ~tie & (n_t == vt) & (n_o == vo)
        rev = have & ~tie & (n_t == vo) & (n_o == vt)
        other = have & ~tie & ~agree & ~rev
        d["cls"] = np.where(~have, "厂商人数缺失", np.where(tie, "平局", np.where(agree, "一致", np.where(rev, "反转", "其他"))))
        for g, h in d.groupby("grp"):
            vc = h["cls"].value_counts()
            n = len(h)
            dec = int(vc.get("一致", 0) + vc.get("反转", 0))
            rows.append({"任务": task, "赛季": g, "带帧事件数": n, "一致%": round(100 * vc.get("一致", 0) / n, 2), "反转%": round(100 * vc.get("反转", 0) / n, 3),
                         "平局%": round(100 * vc.get("平局", 0) / n, 1), "其他%": round(100 * vc.get("其他", 0) / n, 2), "厂商人数缺失%": round(100 * vc.get("厂商人数缺失", 0) / n, 2),
                         "可判定里反转%": round(100 * vc.get("反转", 0) / dec, 3) if dec else float("nan")})
        h = d[d["grp"] == "25/26"]
        for mid, hh in h.groupby("match_id"):
            vc = hh["cls"].value_counts()
            dec = int(vc.get("一致", 0) + vc.get("反转", 0))
            per_match.append({"match_id": mid, "task": task, "n": len(hh), "n_decidable": dec, "n_rev": int(vc.get("反转", 0)), "n_other": int(vc.get("其他", 0)),
                              "rev_pct_decidable": 100 * vc.get("反转", 0) / dec if dec else float("nan")})
        print(f"[{task}] {len(d):,} 事件  ({time.time() - t0:.0f}s)", flush=True)
    tab = pd.DataFrame(rows)
    key = list(zip(tab["任务"], tab["赛季"]))
    if len(set(key)) != len(key):
        raise RuntimeError("汇总表同一个键出现两次")
    tab["赛季"] = pd.Categorical(tab["赛季"], ORDER, ordered=True)
    tab = tab.sort_values(["任务", "赛季"])
    pm = pd.DataFrame(per_match)
    if pm.duplicated(["match_id", "task"]).any():
        raise RuntimeError("逐场表同一个键出现两次")
    OUT_PQ.parent.mkdir(parents=True, exist_ok=True)
    pm.to_parquet(OUT_PQ, index=False)
    piv = pm.pivot(index="match_id", columns="task", values="rev_pct_decidable")
    worst = pm.groupby("match_id").agg(n_decidable=("n_decidable", "sum"), n_rev=("n_rev", "sum"), n_other=("n_other", "sum")).reset_index()
    worst["rev_pct_all5"] = 100 * worst["n_rev"] / worst["n_decidable"]
    L = ["# 留出数据确认，25/26 五个任务的 teammate 标记直接检验", "",
         f"由 `scripts/eval/holdout_2526_flag_check.py` 生成，{time.strftime('%Y-%m-%d %H:%M')}。张量通道 0、1 之和对厂商 visible_player_counts，逐事件分类，不涉及模型。", "",
         "## 按任务和赛季组", "", tab.to_markdown(index=False), "",
         "## 25/26 逐场（五个任务合计，可判定事件里反转的比例）", "",
         f"146 场里五任务合计反转比例最大 {worst['rev_pct_all5'].max():.3f}%，中位 {worst['rev_pct_all5'].median():.3f}%；反转事件总数 {int(worst['n_rev'].sum())} / 可判定 {int(worst['n_decidable'].sum()):,}；“其他”总数 {int(worst['n_other'].sum())}。", "",
         worst.round(3).to_markdown(index=False), "", "## 25/26 逐场逐任务（可判定里反转%）", "", piv.round(2).reset_index().to_markdown(index=False), ""]
    OUT_MD.write_text("\n".join(L), encoding="utf8")
    print(tab.to_string(index=False))
    print(f"[done] {OUT_MD}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
