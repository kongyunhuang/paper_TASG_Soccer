#!/usr/bin/env python3
"""
原始 360 数据全量扫描  逐场逐事件读 StatsBomb 原始文件，落一张比赛清单和一张事件帧诊断表
============================
只读 Google Drive 的 datos 目录，不改任何原始文件。全量扫描，不抽样。
每场比赛产出文件层面的清点（事件数、360 条目数、带帧事件数、文件大小和修改时间、坐标小数位数），
每个事件产出帧的诊断量，供后续脚本判断帧的坐标朝向和 teammate 标记是以哪支球队为准。
  朝向  帧里 actor 到事件位置和到镜像位置 (120-x, 80-y) 的距离
  标记  teammate 为真的点数，和厂商 visible_player_counts 里事件所属球队、对方球队的点数
  门将  标为 keeper 的点的 x 坐标和 teammate 标记
  配对  同一时刻或相邻的对方球队事件里，有没有点集相同（允许整体平移、允许镜像）的帧，标记是否一致
  格内人数  事件所在格（10x10 码）和镜像格里 teammate 为真、为假的点数，以及 3x3 邻域
未下载到本地的文件（只有 25/26 目录里有）跳过不读，避免触发云盘下载，清单里标 local=False。

约定出处  StatsBomb API 360 Frames v2.0.0 第 3 页，location 与所属事件同向（actor 的球队从 0 向 120 进攻），
          teammate 指与 actor 同队。球场 120x80 码。

输入  datos/{赛季目录}/*_events.json、*_360.json、00_info_all_matches_*.json
输出  audit/raw_scan/match_inventory.parquet     每场一行
      audit/raw_scan/match_metadata.parquet      厂商比赛元数据每场一行
      audit/raw_scan/events/{赛季目录}.parquet   每个事件一行

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/audit/scan_raw_360.py --workers 3
  PYTHONPATH=. python -u scripts/audit/scan_raw_360.py --limit 5   （每赛季只跑 5 场，试跑用，输出到 audit/raw_scan_limit5/）

Last modified 2026-10-06 (public release: data directory from STATSBOMB_DATA_DIR; logic unchanged since 2026-10-02)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

# Directory with the raw StatsBomb files, one sub-directory per season. Set the
# STATSBOMB_DATA_DIR environment variable; the default is data/statsbomb under the
# repository root. The data are not distributed with this repository (see README).
DATA_DIR = Path(os.environ.get("STATSBOMB_DATA_DIR", "data/statsbomb"))
SEASONS = ["11_235_2022_23", "11_281_2023_24", "11_317_2024_25", "11_318_2025_26_part",
           "2_235_2022_23", "2_281_2023_24", "2_317_2024_25", "2_318_2025_26_part"]
SPECIAL_KEYS = {"Goal Keeper": "goalkeeper", "Ball Receipt*": "ball_receipt", "50/50": "50_50"}
NAN = float("nan")


def is_local(path: str) -> bool:
    """云盘占位文件的磁盘块数为 0，读它会触发下载"""
    st = os.stat(path)
    return st.st_blocks * 512 >= st.st_size * 0.9


def ts_seconds(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def type_detail(e):
    t = e["type"]["name"]
    d = e.get(SPECIAL_KEYS.get(t, t.lower().replace(" ", "_"))) or {}
    if not isinstance(d, dict):
        d = {}
    sub = (d.get("type") or {}).get("name") or ""
    out = (d.get("outcome") or {}).get("name") or ""
    aw = bool(d.get("aerial_won"))
    return t, sub, out, aw


def cell(x, y):
    return min(max(int(x / 10), 0), 11), min(max(int(y / 10), 0), 7)


def fit(A, TA, AA, B, TB, AB, mirror):
    """B 经过（可选镜像加）整体平移后是否与 A 点集重合。返回 平移距离、标记一致比例、actor 是否同一点。不重合返回 None"""
    B2 = np.array([120.0, 80.0]) - B if mirror else B
    t = A.mean(0) - B2.mean(0)
    d = np.sqrt(((A[:, None, :] - (B2 + t)[None, :, :]) ** 2).sum(-1))
    if d.min(1).max() > 0.25 or d.min(0).max() > 0.25:
        return None
    m = d.argmin(1)
    actor_same = bool(AA.any() and AB.any() and AB[m[AA.argmax()]])
    return float(np.sqrt((t ** 2).sum())), float((TA == TB[m]).mean()), actor_same


def scan_match(job):
    season, fe = job
    f360 = fe.replace("_events.json", "_360.json")
    mid = int(os.path.basename(fe).split("_")[0])
    inv = dict(season_dir=season, match_id=mid, events_local=is_local(fe),
               events_bytes=os.path.getsize(fe), events_mtime=os.path.getmtime(fe),
               f360_exists=os.path.exists(f360), f360_local=False, f360_bytes=0, f360_mtime=NAN,
               n_360_entries=-1, n_360_nonempty=-1, n_360_orphan=-1, coord_1dec_frac=NAN)
    if not inv["events_local"]:
        return inv, None
    ev = json.load(open(fe, encoding="utf-8"))
    fr = {}
    if inv["f360_exists"]:
        inv["f360_bytes"] = os.path.getsize(f360)
        inv["f360_mtime"] = os.path.getmtime(f360)
        inv["f360_local"] = is_local(f360)
        if inv["f360_local"]:
            frames = json.load(open(f360, encoding="utf-8"))
            fr = {x["event_uuid"]: x for x in frames}
            ids = {e["id"] for e in ev}
            inv["n_360_entries"] = len(frames)
            inv["n_360_nonempty"] = sum(1 for x in frames if x.get("freeze_frame"))
            inv["n_360_orphan"] = sum(1 for x in frames if x["event_uuid"] not in ids)
            vals = [c for x in frames[:300] for p in (x.get("freeze_frame") or []) for c in p["location"]]
            if vals:
                inv["coord_1dec_frac"] = float(np.mean([abs(v * 10 - round(v * 10)) < 1e-6 for v in vals]))

    rows, arrs = [], []
    for i, e in enumerate(ev):
        t, sub, out, aw = type_detail(e)
        loc = e.get("location")
        team = e["team"]["id"]
        r = dict(season_dir=season, match_id=mid, event_id=e["id"], idx=e["index"], period=e["period"],
                 ts=ts_seconds(e["timestamp"]), type=t, subtype=sub, outcome=out, aerial_won=aw,
                 team_id=team, poss_team_id=e["possession_team"]["id"],
                 player_id=(e.get("player") or {}).get("id", -1),
                 loc_x=loc[0] if loc else NAN, loc_y=loc[1] if loc else NAN,
                 uuid_ver=e["id"][14], under_pressure=bool(e.get("under_pressure")),
                 has_entry=False, has_frame=False)
        A = None
        f = fr.get(e["id"])
        if f is not None:
            r["has_entry"] = True
            ff = f.get("freeze_frame") or []
            if ff:
                r["has_frame"] = True
                P = np.array([p["location"][:2] for p in ff], dtype=float)
                T = np.array([bool(p.get("teammate")) for p in ff])
                AC = np.array([bool(p.get("actor")) for p in ff])
                K = np.array([bool(p.get("keeper")) for p in ff])
                A = (P, T, AC)
                r.update(n_pts=len(ff), n_T=int(T.sum()), n_act=int(AC.sum()), n_keep=int(K.sum()),
                         has_visible_area=bool(f.get("visible_area")),
                         sb_ngs=f.get("num_defenders_on_goal_side_of_actor"),
                         sb_dnd=f.get("distance_to_nearest_defender"))
                vt, vo = -1, -1
                for d in f.get("visible_player_counts") or []:
                    if d.get("team_id") == team:
                        vt = d.get("count", -1)
                    else:
                        vo = d.get("count", -1)
                r["vpc_team"], r["vpc_other"] = vt, vo
                kT, kO = P[K & T, 0], P[K & ~T, 0]
                r["kT_x"] = float(kT[0]) if len(kT) else NAN
                r["kO_x"] = float(kO[0]) if len(kO) else NAN
                if AC.any():
                    a = P[AC][0]
                    r.update(act_x=float(a[0]), act_y=float(a[1]), act_T=bool(T[AC][0]))
                    oth = ~AC
                    dA = np.sqrt(((P - a) ** 2).sum(1))
                    r["dT_act"] = float(dA[T & oth].min()) if (T & oth).any() else NAN
                    r["dO_act"] = float(dA[~T & oth].min()) if (~T & oth).any() else NAN
                    r["gT_act_gt"] = int((P[T & oth, 0] > a[0]).sum())
                    r["gT_act_lt"] = int((P[T & oth, 0] < a[0]).sum())
                    r["gO_act_gt"] = int((P[~T & oth, 0] > a[0]).sum())
                    r["gO_act_lt"] = int((P[~T & oth, 0] < a[0]).sum())
                if loc:
                    gx, gy = cell(loc[0], loc[1])
                    cx = np.clip((P[:, 0] / 10).astype(int), 0, 11)
                    cy = np.clip((P[:, 1] / 10).astype(int), 0, 7)
                    mx = np.clip(((120 - P[:, 0]) / 10).astype(int), 0, 11)
                    my = np.clip(((80 - P[:, 1]) / 10).astype(int), 0, 7)
                    s0, m0 = (cx == gx) & (cy == gy), (mx == gx) & (my == gy)
                    s3, m3 = (abs(cx - gx) <= 1) & (abs(cy - gy) <= 1), (abs(mx - gx) <= 1) & (abs(my - gy) <= 1)
                    r.update(cT_s=int((s0 & T).sum()), cO_s=int((s0 & ~T).sum()),
                             cT_m=int((m0 & T).sum()), cO_m=int((m0 & ~T).sum()),
                             nT_s=int((s3 & T).sum()), nO_s=int((s3 & ~T).sum()),
                             nT_m=int((m3 & T).sum()), nO_m=int((m3 & ~T).sum()))
        rows.append(r)
        arrs.append(A)

    # 配对  同一时刻或索引相差不超过 4 的对方球队事件里找点集相同的帧
    n = len(ev)
    for i in range(n):
        if arrs[i] is None:
            continue
        P, T, AC = arrs[i]
        ri = rows[i]
        cands = []
        for j in range(max(0, i - 12), min(n, i + 13)):
            if j == i or arrs[j] is None:
                continue
            rj = rows[j]
            if rj["team_id"] == ri["team_id"] or rj["period"] != ri["period"]:
                continue
            dt = abs(rj["ts"] - ri["ts"])
            if dt > 1e-9 and abs(j - i) > 4:
                continue
            cands.append((dt, abs(j - i), j))
        cands.sort()
        ri["pair_n_cand"] = len(cands)
        for dt, _, j in cands:
            B, TB, AB = arrs[j]
            if len(B) != len(P):
                continue
            hit = None
            for rel, mirror in (("S", False), ("M", True)):
                got = fit(P, T, AC, B, TB, AB, mirror)
                if got is not None:
                    hit = (rel,) + got
                    break
            if hit:
                rj = rows[j]
                ri.update(pair_rel=hit[0], pair_trans=hit[1], pair_flag_agree=hit[2], pair_actor_same=hit[3],
                          pair_type=rj["type"], pair_subtype=rj["subtype"], pair_outcome=rj["outcome"],
                          pair_aerial_won=rj["aerial_won"], pair_didx=rj["idx"] - ri["idx"], pair_dt=dt,
                          pair_event_id=rj["event_id"])
                break
    return inv, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    out = Path(f"audit/raw_scan_limit{a.limit}" if a.limit else "audit/raw_scan")
    (out / "events").mkdir(parents=True, exist_ok=True)

    meta = []
    for s in SEASONS:
        for p in (DATA_DIR / s).glob("00_info_all_matches_*.json"):
            if not is_local(str(p)):
                print(f"[meta] {p.name} 未下载到本地，跳过", flush=True)
                continue
            st = os.stat(p)
            for m in json.load(open(p, encoding="utf-8")):
                md = m.get("metadata") or {}
                meta.append(dict(season_dir=s, match_id=int(m["match_id"]), match_date=str(m.get("match_date")),
                                 match_week=m.get("match_week"), collection_status=m.get("collection_status"),
                                 play_status=m.get("play_status"), match_status=m.get("match_status"),
                                 match_status_360=m.get("match_status_360"), last_updated=m.get("last_updated"),
                                 last_updated_360=m.get("last_updated_360"), data_version=md.get("data_version"),
                                 shot_fidelity_version=md.get("shot_fidelity_version"),
                                 xy_fidelity_version=md.get("xy_fidelity_version"),
                                 info_file_mtime=st.st_mtime, n_matches_in_info=0))
    meta = pd.DataFrame(meta)
    meta.to_parquet(out / "match_metadata.parquet", index=False)
    print(f"[meta] {len(meta)} 场比赛的元数据已保存到 {out / 'match_metadata.parquet'}", flush=True)

    jobs_by_season = {}
    for s in SEASONS:
        files = sorted(str(p) for p in (DATA_DIR / s).glob("*_events.json"))
        if a.limit:
            files = files[:a.limit]
        jobs_by_season[s] = [(s, f) for f in files]
    total = sum(len(v) for v in jobs_by_season.values())
    print(f"[start] {total} 场, workers={a.workers}", flush=True)

    t0, done, invs = time.time(), 0, []
    with Pool(a.workers) as pool:
        for s, jobs in jobs_by_season.items():
            dfs = []
            for inv, df in pool.imap_unordered(scan_match, jobs, chunksize=2):
                invs.append(inv)
                if df is not None:
                    dfs.append(df)
                done += 1
                if done % 50 == 0 or done == total:
                    el = time.time() - t0
                    print(f"[{done}/{total}] {s}  已用 {el / 60:.1f} 分  预计剩余 {el / done * (total - done) / 60:.0f} 分", flush=True)
            if dfs:
                d = pd.concat(dfs, ignore_index=True)
                d.to_parquet(out / "events" / f"{s}.parquet", index=False)
                print(f"[save] {s}  {len(dfs)} 场  {len(d):,} 个事件  带帧 {int(d['has_frame'].sum()):,}"
                      f" ({100 * d['has_frame'].mean():.1f}%)  -> {out / 'events' / (s + '.parquet')}", flush=True)
                del d, dfs
    pd.DataFrame(invs).to_parquet(out / "match_inventory.parquet", index=False)
    print(f"[done] {(time.time() - t0) / 60:.1f} 分  清单 {out / 'match_inventory.parquet'}", flush=True)


if __name__ == "__main__":
    main()
