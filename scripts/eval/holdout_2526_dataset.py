#!/usr/bin/env python3
"""
留出数据确认，数据构造  给 25/26 的留出比赛按 train_unified_clean 完全相同的口径组特征、标签和空间张量
============================
train_unified_clean.build_dataset 写死了 FULL_SEASONS 和 24/25 的状态表，导入它拿不到 25/26。本文件新写一个构造函数
build_holdout，特征列、标签定义、张量查找、过滤条件逐行照搬 build_dataset，只是赛季和场次由参数给，并且不抽样（留出集全量用）。
留出任务只有 pass、dest、shot、interception、ball_recovery、pressure 六个。dribble 和 tackle 的摆正版张量不含 25/26，不在这里构造。
  --verify  在 24/25 的 202 场干净比赛上核对。先用原 build_dataset 组出测试集（抽样固定），再用 build_holdout 组同一批比赛，
            按 event_id 对齐后要求 X、y、sm、match 逐位相同，并且原测试集的 event_id 全部在本构造里。核对通过才允许用于 25/26。
  默认      只打印 25/26 留出集各任务的条数和标签均值，不保存。张量按需 mmap 读取。
不写 u1_clean、single_clean、single_clean_rich。核对结果写到 audit/holdout_2526_dataset_verify.md。

输入  同 train_unified_clean.py，另读 data/cache/holdout_2526/clean_matches_2526.parquet
输出  audit/holdout_2526_dataset_verify.md（--verify 时）

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/eval/holdout_2526_dataset.py --verify
  PYTHONPATH=. python -u scripts/eval/holdout_2526_dataset.py

Last modified 2026-10-02
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.training.train_all import CACHE_DIR, TEST_SEASONS
from scripts.training.train_unified_clean import F_BASE, F_PASS, F_SB, pressure_labels_5s

HOLDOUT_SEASONS = ["11_318_2025_26_part", "2_318_2025_26_part"]
HOLDOUT_TASKS = ["pass", "dest", "shot", "interception", "ball_recovery", "pressure"]
HOLDOUT_DIR = CACHE_DIR / "holdout_2526"
VERIFY_MD = Path("audit/holdout_2526_dataset_verify.md")
SRC = {"pass": "pass", "dest": "pass", "shot": "shot", "interception": "interception", "ball_recovery": "ball_recovery", "pressure": "pressure"}
TYPE_NAME = {"pass": "Pass", "shot": "Shot", "interception": "Interception", "ball_recovery": "Ball Recovery", "pressure": "Pressure"}


def holdout_match_ids():
    return set(pd.read_parquet(HOLDOUT_DIR / "clean_matches_2526.parquet")["match_id"].astype(int))


def load_events(seasons, matches=None):
    """读事件表并做和 build_dataset 相同的预处理（只多了按场次过滤）。返回的表含这些比赛的全部事件类型，供 5 秒标签用。"""
    cols = ["event_id", "match_id", "period", "timestamp", "team_id", "type_name", "season_dir"] + F_BASE + F_PASS + F_SB + \
           ["pass_end_location_x", "pass_end_location_y", "pass_outcome_name", "shot_outcome_name", "shot_type_name",
            "interception_outcome_name", "ball_recovery_failure"]
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet", columns=cols)
    df = df[df["season_dir"].isin(seasons)]
    if matches is not None:
        df = df[df["match_id"].astype(int).isin(matches)]
    df = df.reset_index(drop=True)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    for c in ["pass_is_progressive", "pass_cross", "pass_switch", "pass_through_ball", "pass_cut_back"]:
        df[c] = df[c].fillna(False).astype(int)
    return df


def build_task(df, press_y, task):
    """一个任务的全量留出集。过滤、标签、特征和 build_dataset 逐行相同，不抽样。"""
    zeros_pass = np.zeros(len(F_PASS), dtype=np.float32)
    src = SRC[task]
    d = df[df["type_name"] == TYPE_NAME[src]]
    if task == "shot":
        d = d[d["shot_type_name"] != "Penalty"]
    if task == "dest":
        d = d[d["pass_end_location_x"].notna() & d["pass_end_location_y"].notna()]
    sidx = pd.read_parquet(CACHE_DIR / f"action_soccermaps_{src}_idx.parquet")
    lookup = {e: i for i, e in enumerate(sidx["event_id"].values)}
    d = d[d["event_id"].isin(lookup)].reset_index(drop=True)
    smaps = np.load(CACHE_DIR / f"action_soccermaps_{src}.npy", mmap_mode="r")
    base = d[F_BASE].fillna(0).values.astype(np.float32)
    ps = d[F_PASS].fillna(0).values.astype(np.float32) if task == "pass" else np.tile(zeros_pass, (len(d), 1))
    sb = d[F_SB].fillna(0).values.astype(np.float32)
    X = np.hstack([base, ps, sb])
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
    rows = np.array([lookup[e] for e in d["event_id"].values])
    order = np.argsort(rows)
    sm = np.empty((len(rows), 7, 8, 12), dtype=np.float32)
    sm[order] = np.asarray(smaps[rows[order]])
    return dict(X=X, y=y, sm=sm, match=d["match_id"].values.astype(np.int64), eid=d["event_id"].values.astype(str))


def build_holdout(tasks=HOLDOUT_TASKS, seasons=HOLDOUT_SEASONS, matches=None, verbose=True):
    """返回 {task: dict(X, y, sm, match, eid)}，默认是 25/26 通过代理口径的全部比赛、全量事件。"""
    t0 = time.time()
    if matches is None and seasons == HOLDOUT_SEASONS:
        matches = holdout_match_ids()
    df = load_events(seasons, matches)
    press_y = pressure_labels_5s(df)
    out = {}
    for task in tasks:
        out[task] = build_task(df, press_y, task)
        if verbose:
            print(f"[holdout] {task:14s} n {len(out[task]['y']):7,}  matches {len(np.unique(out[task]['match'])):4d}  "
                  f"mean label {out[task]['y'].mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def verify():
    """在 24/25 的 202 场干净比赛上核对 build_holdout 和 build_dataset 的测试集逐位相同。"""
    from scripts.eval.summarize_u1_clean import clean_matches
    from scripts.training.train_unified_clean import build_dataset
    t0 = time.time()
    clean = clean_matches()
    orig = build_dataset()
    for t in list(orig):
        orig[t].pop("dev", None)  # 只留测试集，省内存
    print(f"[verify] 原 build_dataset 测试集组好  ({time.time() - t0:.0f}s)", flush=True)
    L = ["# 留出数据构造的核对（24/25 的 202 场干净比赛）", "",
         f"由 `scripts/eval/holdout_2526_dataset.py --verify` 生成，{time.strftime('%Y-%m-%d %H:%M')}。",
         "原 build_dataset 的测试集（抽样固定）限制到 202 场后，和 build_holdout 在同一批比赛上的全量构造按 event_id 对齐，逐位比较。", "",
         "| 任务 | 原测试集在 202 场内的条数 | 本构造的条数 | 原 event_id 全在本构造里 | X 逐位相同 | y 相同 | 张量逐位相同 | match 相同 | 结论 |", "|---|---:|---:|---|---|---|---|---|---|"]
    all_ok = True
    df = load_events(TEST_SEASONS, clean)
    press_y = pressure_labels_5s(df)
    for t in HOLDOUT_TASKS:
        o = orig[t]["test"]
        keep = np.array([int(m) in clean for m in o["match"]])
        o_eid = o["eid"][keep]
        m = build_task(df, press_y, t)  # 逐任务构造，省内存
        pos = pd.Series(np.arange(len(m["eid"])), index=m["eid"])
        subset = bool(pd.Index(o_eid).isin(pos.index).all())
        if not subset:
            all_ok = False
            L.append(f"| {t} | {int(keep.sum()):,} | {len(m['y']):,} | 否 | 未比 | 未比 | 未比 | 未比 | **不通过** |")
            continue
        j = pos.loc[o_eid].values
        x_ok = bool(np.array_equal(o["X"][keep], m["X"][j]))
        y_ok = bool(np.array_equal(o["y"][keep], m["y"][j]))
        sm_ok = bool(np.array_equal(o["sm"][keep], m["sm"][j]))
        mt_ok = bool(np.array_equal(o["match"][keep], m["match"][j]))
        ok = x_ok and y_ok and sm_ok and mt_ok
        all_ok &= ok
        L.append(f"| {t} | {int(keep.sum()):,} | {len(m['y']):,} | 是 | {'是' if x_ok else '否'} | {'是' if y_ok else '否'} | {'是' if sm_ok else '否'} | {'是' if mt_ok else '否'} | {'通过' if ok else '**不通过**'} |")
        print(f"[verify] {t:14s} orig {int(keep.sum()):7,} mine {len(m['y']):7,} subset {subset} X {x_ok} y {y_ok} sm {sm_ok} match {mt_ok}", flush=True)
        del m, orig[t]
    L += ["", f"总结论  {'全部通过' if all_ok else '有不通过项，不得用于 25/26'}。", ""]
    VERIFY_MD.write_text("\n".join(L), encoding="utf8")
    print(f"[verify] {'全部通过' if all_ok else '有不通过项'}  {VERIFY_MD}  ({time.time() - t0:.0f}s)", flush=True)
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        verify()
    else:
        build_holdout()
