#!/usr/bin/env python3
"""
原始 360 数据核查的证据表  从全量扫描结果算出赛季完整性、帧视角是否符合厂商约定、摆正前后的泄漏指标
============================
读 scan_raw_360.py 的输出，不再碰原始文件。分七部分。
  T1  各赛季的比赛数、状态、版本字段、更新时间、文件时间、每场事件数和带帧比例
  T2  23/24 和 24/25 按比赛月份的带帧比例（全部事件和射门），带帧比例最低的比赛，
      24/25 每场比赛处于标记反转还是正常状态、与厂商 last_updated_360 和射门带帧率的关系
  T3  项目缓存 L1_events_v3.parquet 与云盘原始文件逐赛季对账
  T4  全部事件类别的帧是否符合说明书（坐标朝向、teammate 标记），三条独立规则互相校验
  T5  不合规帧的来源，同一时刻对方球队事件的帧与本帧的关系，以及与 team、possession_team 字段的关系
  T6  Dribble、Ball Receipt、Duel 的泄漏指标在摆正前后的对比
  T7  逼抢任务的正例率和 possession_team 切换频率，按赛季和 24/25 场次状态（只用事件文件，与帧无关）

判定规则（均不使用结果标签）
  坐标镜像  帧里 actor 离镜像位置 (120-x, 80-y) 比离事件位置更近
  标记反转  vpc 规则  teammate 为真的点数等于厂商 visible_player_counts 里对方球队的点数且不等于本方点数
            门将规则  坐标摆正后，靠近球门的门将所在半场与 teammate 标记对不上
            配对规则  同一时刻对方球队事件的帧与本帧点集相同且 teammate 标记逐点相同
约定出处  StatsBomb API 360 Frames v2.0.0 第 3 页

输入  audit/raw_scan/match_inventory.parquet、match_metadata.parquet、events/*.parquet
      data/cache/L1_events_v3.parquet
输出  audit/raw_360_check_tables.md
      audit/raw_scan/corrected_flags_dribble_duel_br.parquet   三类任务事件逐条的判定结果
      audit/raw_scan/match_state_2425.parquet                  24/25 每场比赛的状态

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/audit/analyze_raw_360.py

Last modified 2026-10-02
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

SEASONS = ["11_235_2022_23", "11_281_2023_24", "11_317_2024_25", "11_318_2025_26_part",
           "2_235_2022_23", "2_281_2023_24", "2_317_2024_25", "2_318_2025_26_part"]
DICT_COLS = ["season_dir", "type", "subtype", "outcome", "uuid_ver", "pair_rel", "pair_type", "pair_subtype", "pair_outcome"]
TACKLE_WIN = {"Won", "Success", "Success In Play", "Success Out"}


def md(df, index=False):
    return df.to_markdown(index=index)


def season_label(s):
    lg = "La Liga" if s.startswith("11_") else "EPL"
    yr = {"2022_23": "22/23", "2023_24": "23/24", "2024_25": "24/25", "2025_26": "25/26"}[[k for k in ("2022_23", "2023_24", "2024_25", "2025_26") if k in s][0]]
    return f"{lg} {yr}"


def load_events(scan: Path):
    dfs = []
    for s in SEASONS:
        p = scan / "events" / f"{s}.parquet"
        if not p.exists():
            continue
        cols = [c for c in pq.ParquetFile(p).schema.names if c not in ("pair_event_id",)]
        t = pq.read_table(p, columns=cols, read_dictionary=[c for c in DICT_COLS if c in cols])
        d = t.to_pandas()
        for c in d.columns:
            if d[c].dtype == "float64":
                d[c] = d[c].astype("float32")
        dfs.append(d)
        print(f"[load] {s} {len(d):,}", flush=True)
    d = pd.concat(dfs, ignore_index=True)
    for c in DICT_COLS:
        if c in d:
            d[c] = d[c].astype("category")
    return d


def add_derived(d):
    s = d["season_dir"].astype(str)
    d["season"] = s.map(season_label).astype("category")
    d["era"] = np.where(s.str.contains("2025_26"), "25/26", np.where(s.str.contains("2024_25"), "24/25", "22/23 and 23/24"))
    t, sub, out = d["type"].astype(str), d["subtype"].astype(str), d["outcome"].astype(str)
    cls = t.copy()
    m = t == "Dribble"
    cls[m] = "Dribble " + out[m]
    m = t == "Ball Receipt*"
    cls[m] = np.where(out[m] == "", "Ball Receipt Complete", "Ball Receipt Incomplete")
    m = t == "Duel"
    cls[m] = "Duel " + sub[m]
    m = t == "50/50"
    cls[m] = "50/50 " + out[m]
    m = d["aerial_won"] & t.isin(["Pass", "Clearance", "Shot", "Miscontrol"])
    cls[m] = t[m] + " (aerial won)"
    d["cls"] = cls.astype("category")
    ok = d["has_frame"] & d["loc_x"].notna() & (d["n_act"] > 0)
    d_s = np.hypot(d["act_x"] - d["loc_x"], d["act_y"] - d["loc_y"])
    d_m = np.hypot(d["act_x"] - (120 - d["loc_x"]), d["act_y"] - (80 - d["loc_y"]))
    d["d_s"], d["d_m"] = d_s, d_m
    d["usable"] = ok
    d["mir"] = ok & (d_m < d_s)
    # vpc 规则
    f = np.full(len(d), "na", dtype=object)
    has = ok & (d["vpc_team"] >= 0) & (d["vpc_other"] >= 0)
    tie = has & (d["vpc_team"] == d["vpc_other"])
    f[has & ~tie & (d["n_T"] == d["vpc_team"])] = "ok"
    f[has & ~tie & (d["n_T"] == d["vpc_other"])] = "rev"
    f[has & ~tie & (d["n_T"] != d["vpc_team"]) & (d["n_T"] != d["vpc_other"])] = "other"
    f[tie] = "tie"
    d["vpc_rule"] = pd.Categorical(f)
    # 门将规则  坐标摆正后看靠近球门的门将
    kT = np.where(d["mir"], 120 - d["kT_x"], d["kT_x"])
    kO = np.where(d["mir"], 120 - d["kO_x"], d["kO_x"])
    vT = ~np.isnan(kT) & ((kT < 30) | (kT > 90))
    vO = ~np.isnan(kO) & ((kO < 30) | (kO > 90))
    bad = (vT & (kT > 60)).astype(int) + (vO & (kO < 60)).astype(int)
    good = (vT & (kT < 60)).astype(int) + (vO & (kO > 60)).astype(int)
    k = np.full(len(d), "na", dtype=object)
    k[ok & (good > 0) & (bad == 0)] = "ok"
    k[ok & (bad > 0) & (good == 0)] = "rev"
    k[ok & (bad > 0) & (good > 0)] = "mixed"
    d["keeper_rule"] = pd.Categorical(k)
    # 配对规则
    p = np.full(len(d), "na", dtype=object)
    hp = ok & d["pair_rel"].notna()
    p[hp & (d["pair_flag_agree"] > 0.9)] = "same_flags"
    p[hp & (d["pair_flag_agree"] < 0.1)] = "opposite_flags"
    p[hp & (d["pair_flag_agree"] >= 0.1) & (d["pair_flag_agree"] <= 0.9)] = "partial"
    d["pair_rule"] = pd.Categorical(p)
    return d


def pct(x):
    return round(100 * float(np.mean(x)), 1) if len(x) else np.nan


# ── T1 T2 T3 ─────────────────────────────────────────────────────────

def t1_inventory(d, inv, meta, L):
    inv = inv.copy()
    inv["season"] = inv["season_dir"].map(season_label)
    meta = meta.copy()
    meta["season"] = meta["season_dir"].map(season_label)
    g = d.groupby(["season_dir", "match_id"], observed=True).agg(n_events=("idx", "size"), n_frame=("has_frame", "sum"),
                                                                 n_entry=("has_entry", "sum")).reset_index()
    g["season"] = g["season_dir"].astype(str).map(season_label)
    g["frame_pct"] = 100 * g["n_frame"] / g["n_events"]
    inv2 = inv.merge(g[["match_id", "n_events", "n_frame", "frame_pct"]], on="match_id", how="left")
    rows = []
    for s in [season_label(x) for x in SEASONS]:
        m, i = meta[meta["season"] == s], inv2[inv2["season"] == s]
        full = i[i["f360_local"]]
        rows.append({
            "赛季": s,
            "元数据列出的比赛": len(m),
            "事件状态 available": int((m["match_status"] == "available").sum()),
            "360 状态 available": int((m["match_status_360"] == "available").sum()),
            "collection_status Complete": int((m["collection_status"] == "Complete").sum()),
            "云盘 events 文件": len(i),
            "云盘 360 文件": int(i["f360_exists"].sum()),
            "两个文件都在本地并已扫描": len(full),
            "每场事件数 均值 (最小 到 最大)": f"{full['n_events'].mean():.0f} ({full['n_events'].min():.0f} 到 {full['n_events'].max():.0f})" if len(full) else "未扫描",
            "每场 360 条目 均值 (最小 到 最大)": f"{full['n_360_entries'].mean():.0f} ({full['n_360_entries'].min()} 到 {full['n_360_entries'].max()})" if len(full) else "未扫描",
            "带帧事件占比 均值 (最小 到 最大)": f"{full['frame_pct'].mean():.1f}% ({full['frame_pct'].min():.1f} 到 {full['frame_pct'].max():.1f})" if len(full) else "未扫描",
        })
    L += ["### T1a 各赛季比赛数、状态和每场规模", "",
          "来源  `00_info_all_matches_*.json`（厂商比赛元数据）和逐场 `*_events.json`、`*_360.json` 的全量扫描。带帧指 360 文件里有该事件的条目且 freeze_frame 非空。",
          "", md(pd.DataFrame(rows)), ""]

    rows = []
    for s in [season_label(x) for x in SEASONS]:
        m, i = meta[(meta["season"] == s) & (meta["match_status"] == "available")], inv[inv["season"] == s]
        lu, lu3 = pd.to_datetime(m["last_updated"]), pd.to_datetime(m["last_updated_360"])
        em = pd.to_datetime(i["events_mtime"], unit="s", utc=True).dt.tz_convert("Europe/Madrid")
        ver = d.loc[d["season_dir"].astype(str).map(season_label) == s, "uuid_ver"].astype(str).value_counts(normalize=True)
        rows.append({
            "赛季": s,
            "比赛日期": f"{m['match_date'].min()} 到 {m['match_date'].max()}",
            "data_version": "/".join(sorted(m["data_version"].dropna().unique())),
            "shot_fidelity": "/".join(sorted(m["shot_fidelity_version"].dropna().unique())),
            "xy_fidelity": "/".join(sorted(m["xy_fidelity_version"].dropna().unique())),
            "last_updated 范围": f"{lu.min():%Y-%m-%d} 到 {lu.max():%Y-%m-%d}",
            "last_updated_360 范围": f"{lu3.min():%Y-%m-%d} 到 {lu3.max():%Y-%m-%d}",
            "事件与 360 更新时间相同的场次占比": f"{100 * (m['last_updated'] == m['last_updated_360']).mean():.0f}%",
            "本地文件修改时间（马德里时间）": f"{em.min():%Y-%m-%d %H:%M} 到 {em.max():%Y-%m-%d %H:%M}",
            "事件编号 UUID 版本": " ".join(f"v{k} {100 * v:.0f}%" for k, v in ver.items() if v > 0),
            "帧坐标只有一位小数的比例": f"{100 * i['coord_1dec_frac'].mean():.0f}%" if i["coord_1dec_frac"].notna().any() else "未扫描",
        })
    L += ["### T1b 版本字段、更新时间和文件时间", "",
          "更新时间取自厂商元数据（只算 available 的比赛）。本地文件修改时间是 events 文件在云盘挂载目录里的 mtime，可当作数据拉取时间的上界。",
          "帧坐标小数位数取每场前 300 个帧的坐标值统计。",
          "", md(pd.DataFrame(rows)), ""]
    return inv2


def t2_by_month(d, meta, inv2, L):
    dm = meta.set_index("match_id")["match_date"]
    x = d[d["season_dir"].astype(str).str.contains("2024_25|2023_24")].copy()
    x["month"] = x["match_id"].map(dm).str[:7]
    x["is_shot"] = x["type"].astype(str) == "Shot"
    sec = x["cls"].astype(str).isin(["Dribble Incomplete", "Duel Aerial Lost", "Ball Receipt Incomplete", "Foul Won", "Dispossessed", "Dribbled Past"])
    x["sec"] = sec
    rows = []
    for (mo), g in x.groupby("month"):
        u = g[g["usable"] & g["sec"]]
        dec = u[u["vpc_rule"].isin(["ok", "rev"])]
        rows.append({"比赛月份": mo, "场次": g["match_id"].nunique(), "事件数": len(g),
                     "全部事件带帧%": pct(g["has_frame"]),
                     "射门数": int(g["is_shot"].sum()), "射门带帧%": pct(g.loc[g["is_shot"], "has_frame"]),
                     "帧坐标一位小数%": round(100 * inv2[inv2["match_id"].isin(g["match_id"].unique())]["coord_1dec_frac"].mean(), 0),
                     "UUID v5%": pct(g["uuid_ver"].astype(str) == "5"),
                     "从属事件 坐标镜像%": pct(u["mir"]),
                     "从属事件 标记反转%（vpc 可判定的）": pct(dec["vpc_rule"] == "rev")})
    L += ["### T2a 23/24 和 24/25 按比赛月份（两个联赛合并）", "",
          "从属事件指失败的过人、Aerial Lost、未完成的接球、Foul Won、Dispossessed、Dribbled Past 六类（定义见 T5）。",
          "", md(pd.DataFrame(rows)), ""]
    low = inv2[inv2["f360_local"]].sort_values("frame_pct").head(12).merge(meta[["match_id", "match_date", "last_updated_360"]], on="match_id", how="left")
    L += ["### T2b 带帧事件占比最低的 12 场（全部赛季）", "",
          md(low[["season", "match_id", "match_date", "n_events", "n_360_entries", "frame_pct", "last_updated_360"]].round(1)), ""]


def t2_state_2425(d, meta, scan, L):
    """24/25 每场比赛的状态  用与三个任务标签无关的三类从属事件（Dispossessed、Foul Won、Dribbled Past）判定"""
    x = d[d["era"] == "24/25"]
    v = x[x["usable"] & x["cls"].isin(["Dispossessed", "Foul Won", "Dribbled Past"]) & x["vpc_rule"].isin(["ok", "rev"])]
    pm = v.groupby("match_id").agg(rev_pct=("vpc_rule", lambda s: 100 * (s == "rev").mean()), n=("idx", "size"))
    sh = x[x["type"].astype(str) == "Shot"].groupby("match_id").agg(shots=("idx", "size"), shot_frame_pct=("has_frame", lambda s: 100 * s.mean()))
    def rev_of(c):
        return x[x["usable"] & (x["cls"] == c) & x["vpc_rule"].isin(["ok", "rev"])].groupby("match_id")["vpc_rule"].agg(lambda s: 100 * (s == "rev").mean())
    pm = pm.join(sh)
    pm["aerial_lost_rev_pct"] = rev_of("Duel Aerial Lost")
    pm["br_incomplete_rev_pct"] = rev_of("Ball Receipt Incomplete")
    pm["dribble_incomplete_rev_pct"] = rev_of("Dribble Incomplete")
    m = meta.set_index("match_id")
    pm["match_date"] = m.loc[pm.index, "match_date"]
    pm["last_updated_360"] = pd.to_datetime(m.loc[pm.index, "last_updated_360"])
    pm["league"] = np.where(m.loc[pm.index, "season_dir"].str.startswith("11_"), "La Liga", "EPL")
    pm["state"] = np.where(pm["rev_pct"] >= 80, "反转", np.where(pm["rev_pct"] <= 20, "正常", "中间"))
    pm.reset_index().to_parquet(scan / "match_state_2425.parquet", index=False)

    dist = pd.cut(pm["rev_pct"], [-1, 1, 20, 80, 99, 100], labels=["0 到 1", "1 到 20", "20 到 80", "80 到 99", "99 到 100"]).value_counts().sort_index()
    L += ["### T2c 24/25 每场比赛的状态", "",
          "每场比赛里 Dispossessed、Foul Won、Dribbled Past 三类事件（与论文任务的标签无关）被 vpc 规则判为标记反转的比例。",
          "分布是两头的，没有一场落在 20% 到 80% 之间，所以按 80% 以上记为反转状态、20% 以下记为正常状态。", "",
          md(dist.rename("场次").reset_index().rename(columns={"rev_pct": "每场反转比例%"})), ""]
    rows = []
    for (lg, st), g in pm.groupby(["league", "state"]):
        rows.append({"联赛": lg, "状态": st, "场次": len(g),
                     "比赛日期": f"{g['match_date'].min()} 到 {g['match_date'].max()}",
                     "last_updated_360": f"{g['last_updated_360'].min():%Y-%m-%d} 到 {g['last_updated_360'].max():%Y-%m-%d}",
                     "射门带帧% 场均": round(g["shot_frame_pct"].mean(), 1),
                     "射门带帧超过 85% 的场次": int((g["shot_frame_pct"] > 85).sum()),
                     "Dribble Incomplete 反转% 场均": round(g["dribble_incomplete_rev_pct"].mean(), 1),
                     "Aerial Lost 反转% 场均": round(g["aerial_lost_rev_pct"].mean(), 1),
                     "Ball Receipt Incomplete 反转% 场均": round(g["br_incomplete_rev_pct"].mean(), 1)})
    L += [md(pd.DataFrame(rows)), ""]
    bins = pd.to_datetime(["2024-08-01", "2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-04-15", "2025-05-01", "2025-06-01", "2025-10-01"])
    labels = ["2024-08 到 12", "2025-01", "2025-02", "2025-03", "2025-04-01 到 04-14", "2025-04-15 到 04-30", "2025-05", "2025-06 到 09"]
    ct = pd.crosstab(pd.cut(pm["last_updated_360"], bins, labels=labels), pm["state"])
    L += ["状态与厂商 last_updated_360（该场 360 数据最后一次更新的时间）的关系", "",
          md(ct.reset_index().rename(columns={"last_updated_360": "last_updated_360 所在区间"})), ""]

    v2 = x[x["usable"] & x["vpc_rule"].isin(["ok", "rev"])].copy()
    v2["month"] = v2["match_id"].map(m["match_date"]).str[:7]
    v2["rev"] = (v2["vpc_rule"] == "rev") * 100.0
    sec = ["Dribble Incomplete", "Dribbled Past", "Dispossessed", "Foul Won", "Duel Aerial Lost", "Ball Receipt Incomplete", "50/50 Lost"]
    pv = v2[v2["cls"].isin(sec)].pivot_table(index="cls", columns="month", values="rev", aggfunc="mean", observed=True).round(0).dropna(how="all")
    L += ["### T2d 24/25 各类从属事件的标记反转比例，按比赛月份", "", md(pv.reset_index().rename(columns={"cls": "事件类别"})), ""]

    s = x[(x["type"].astype(str) == "Shot") & (x["subtype"].astype(str) != "Penalty")].copy()
    s["goal"] = s["outcome"].astype(str) == "Goal"
    s["state"] = s["match_id"].map(pm["state"])
    q = s.groupby(["state", "has_frame"]).agg(n=("goal", "size"), goals=("goal", "sum"), goal_rate=("goal", "mean")).round(4).reset_index()
    q.columns = ["场次状态", "射门带帧", "非点球射门数", "进球数", "进球率"]
    L += ["### T2e 24/25 非点球射门的带帧情况和进球率，按场次状态", "", md(q), ""]
    o = s.groupby(["state", s["outcome"].astype(str)]).agg(n=("goal", "size"), frame_pct=("has_frame", lambda z: round(100 * z.mean(), 1))).reset_index()
    o = o.pivot(index="outcome", columns="state", values=["n", "frame_pct"])
    o.columns = [f"{b}状态 {'射门数' if a == 'n' else '带帧%'}" for a, b in o.columns]
    old_s = d[(d["era"] == "22/23 and 23/24") & (d["type"].astype(str) == "Shot") & (d["subtype"].astype(str) != "Penalty")]
    o["22/23 和 23/24 带帧%"] = old_s.groupby(old_s["outcome"].astype(str))["has_frame"].mean().mul(100).round(1)
    L += ["按射门结果看带帧比例", "", md(o.reset_index().rename(columns={"outcome": "射门结果"})), ""]
    return pm


def t3_l1(d, L):
    p = Path("data/cache/L1_events_v3.parquet")
    l1 = pd.read_parquet(p, columns=["season_dir", "match_id", "has_360"])
    a = l1.groupby("season_dir").agg(l1_matches=("match_id", "nunique"), l1_events=("has_360", "size"), l1_with_frame=("has_360", "sum"))
    b = d.groupby("season_dir", observed=True).agg(raw_matches=("match_id", "nunique"), raw_events=("idx", "size"), raw_with_frame=("has_frame", "sum"))
    c = a.join(b, how="outer")
    pm_l1 = l1.groupby("match_id").agg(e=("has_360", "size"), f=("has_360", "sum"))
    pm_raw = d.groupby("match_id").agg(e=("idx", "size"), f=("has_frame", "sum"), s=("season_dir", "first"))
    j = pm_raw.join(pm_l1, lsuffix="_raw", rsuffix="_l1", how="left")
    j["same"] = (j["e_raw"] == j["e_l1"]) & (j["f_raw"] == j["f_l1"])
    c["逐场事件数和带帧数都相同的场次"] = j.groupby("s", observed=True)["same"].sum()
    c.index = [season_label(s) for s in c.index]
    mt = pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC").tz_convert("Europe/Madrid")
    L += ["### T3 项目缓存与云盘原始文件对账", "",
          f"项目目录里没有原始文件的副本，训练用的 `data/cache/L1_events_v3.parquet`（生成于 {mt:%Y-%m-%d %H:%M}）是 `scripts/cache/prepare_l1_events.py` 直接读云盘 datos 生成的。",
          "下表左三列来自缓存，右三列来自本次对云盘原始文件的扫描。25/26 只扫描了已在本地的文件，所以右侧偏少。",
          "", md(c.reset_index().rename(columns={"index": "赛季"})), ""]


# ── T4 T5 ────────────────────────────────────────────────────────────

def t4_compliance(d, L):
    u = d[d["usable"]]
    rows = []
    for (era, cls), g in u.groupby(["era", "cls"], observed=True):
        if len(g) < 200:
            continue
        v = g[g["vpc_rule"].isin(["ok", "rev"])]
        k = g[g["keeper_rule"].isin(["ok", "rev"])]
        rows.append({"时期": era, "事件类别": cls, "带帧事件数": len(g),
                     "坐标镜像%": pct(g["mir"]),
                     "vpc 可判定%": pct(g["vpc_rule"].isin(["ok", "rev"])),
                     "标记反转%（vpc）": pct(v["vpc_rule"] == "rev"),
                     "看得到门将%": pct(g["keeper_rule"].isin(["ok", "rev"])),
                     "标记反转%（门将）": pct(k["keeper_rule"] == "rev"),
                     "actor 恰在事件位置%": pct(g["d_s"] < 0.01)})
    t = pd.DataFrame(rows)
    bad = t[(t["坐标镜像%"] > 5) | (t["标记反转%（vpc）"] > 5)]
    L += ["### T4a 不符合说明书的事件类别（坐标镜像或标记反转超过 5%）", "",
          f"全量统计，带帧、有事件坐标且帧里有 actor 的事件共 {len(u):,} 个（带帧事件共 {int(d['has_frame'].sum()):,} 个）。说明书的约定是坐标与事件同向、teammate 以事件的 actor 为准，合规时三列反常比例都应接近 0。",
          "", md(bad.sort_values(["时期", "带帧事件数"], ascending=[True, False])), "",
          "### T4b 其余事件类别（都合规，每个时期事件数最多的 14 类）", "",
          md(t[~t.index.isin(bad.index)].sort_values(["时期", "带帧事件数"], ascending=[True, False]).groupby("时期").head(14)), ""]

    # 三条规则互相校验
    both = u[u["vpc_rule"].isin(["ok", "rev"]) & u["keeper_rule"].isin(["ok", "rev"])]
    r1 = pd.crosstab(both["vpc_rule"].astype(str), both["keeper_rule"].astype(str))
    bp = u[u["vpc_rule"].isin(["ok", "rev"]) & u["pair_rule"].isin(["same_flags", "opposite_flags"])]
    r2 = pd.crosstab([bp["era"], bp["vpc_rule"].astype(str)], bp["pair_rule"].astype(str))
    L += ["### T4c 三条判定规则的一致性", "",
          "vpc 规则对门将规则（两条都能判定的事件，全部类别全部赛季）", "", md(r1.reset_index()), "",
          f"一致率 {100 * (both['vpc_rule'].astype(str) == both['keeper_rule'].astype(str)).mean():.2f}%，样本 {len(both):,}。", "",
          "vpc 规则对配对规则（有对方配对帧且 vpc 可判定的事件）。same_flags 是本帧和对方事件的帧逐点标记相同，按约定两队的帧标记应当相反。", "",
          md(r2.reset_index()), ""]
    return bad


def t5_mechanism(d, bad, L):
    u = d[d["usable"] & (d["era"] != "25/26")]
    sec = sorted(set(bad["事件类别"]))
    rows = []
    for (era, cls), g in u[u["cls"].isin(sec)].groupby(["era", "cls"], observed=True):
        if len(g) < 200:
            continue
        hp = g["pair_rel"].notna()
        top = g.loc[hp, "pair_type"].astype(str).value_counts(normalize=True).head(3)
        rows.append({"时期": era, "事件类别": cls, "n": len(g),
                     "team 等于 possession_team%": pct(g["team_id"] == g["poss_team_id"]),
                     "有对方配对帧%": pct(hp),
                     "配对帧同时刻%": pct(g.loc[hp, "pair_dt"] < 1e-6),
                     "配对事件类型（前三）": "，".join(f"{k} {100 * v:.0f}%" for k, v in top.items()),
                     "与配对帧坐标同向（未镜像）%": pct(g.loc[hp, "pair_rel"].astype(str) == "S"),
                     "与配对帧标记相同%": pct(g.loc[hp, "pair_flag_agree"] > 0.9),
                     "actor 与配对帧是同一点%": pct(g.loc[hp, "pair_actor_same"].astype(bool)),
                     "没有配对帧的事件里坐标镜像%": pct(g.loc[~hp, "mir"]),
                     })
    L += ["### T5a 不合规事件的帧从哪来", "",
          "配对帧指同一时刻（或索引相差 4 以内）对方球队的事件里，点集与本帧完全重合的帧（允许整体镜像和平移）。",
          "按说明书，同一快照给两队各出一帧时，两帧应当坐标互为镜像、标记逐点相反。",
          "", md(pd.DataFrame(rows).sort_values(["时期", "n"], ascending=[True, False])), ""]

    # 对照  合规的对方事件
    ctrl = ["Pressure", "Duel Tackle", "Foul Committed", "Interception", "Dribble Complete", "Block", "Ball Recovery"]
    rows = []
    for (era, cls), g in u[u["cls"].isin(ctrl)].groupby(["era", "cls"], observed=True):
        hp = g["pair_rel"].notna()
        rows.append({"时期": era, "事件类别": cls, "n": len(g),
                     "team 等于 possession_team%": pct(g["team_id"] == g["poss_team_id"]),
                     "坐标镜像%": pct(g["mir"]),
                     "标记反转%（vpc）": pct(g.loc[g["vpc_rule"].isin(["ok", "rev"]), "vpc_rule"] == "rev"),
                     "有对方配对帧%": pct(hp),
                     "与配对帧坐标互为镜像%": pct(g.loc[hp, "pair_rel"].astype(str) == "M"),
                     "与配对帧标记相同%": pct(g.loc[hp, "pair_flag_agree"] > 0.9)})
    L += ["### T5b 对照，合规的事件类别与 possession_team 的关系", "",
          "如果帧的视角跟的是 possession_team，那么 team 不等于 possession_team 的事件（逼抢、铲球、犯规、拦截）就应当整体镜像，实际没有。",
          "", md(pd.DataFrame(rows).sort_values(["时期", "n"], ascending=[True, False])), ""]

    # 在不合规类别内部，按 team 是否等于 possession_team 分组
    rows = []
    for (era, cls), g in u[u["cls"].isin(sec)].groupby(["era", "cls"], observed=True):
        if len(g) < 200:
            continue
        for same, h in g.groupby(g["team_id"] == g["poss_team_id"]):
            v = h[h["vpc_rule"].isin(["ok", "rev"])]
            rows.append({"时期": era, "事件类别": cls, "team 等于 possession_team": "是" if same else "否", "n": len(h),
                         "坐标镜像%": pct(h["mir"]), "标记反转%（vpc）": pct(v["vpc_rule"] == "rev")})
    L += ["### T5c 不合规类别内部按 team 是否等于 possession_team 分组", "",
          md(pd.DataFrame(rows).sort_values(["时期", "事件类别"])), ""]

    # 配对关系的方向  哪一方的帧是原样的
    p = u[u["pair_rel"].notna() & (u["pair_dt"] < 1e-6) & (u["pair_flag_agree"] > 0.9) & u["vpc_rule"].isin(["ok", "rev"])]
    q = p.groupby(["era", "cls", "pair_type"], observed=True).agg(n=("idx", "size"), rev=("vpc_rule", lambda s: 100 * (s == "rev").mean())).reset_index()
    q = q[q["n"] >= 300].sort_values(["era", "n"], ascending=[True, False]).round(1)
    q.columns = ["时期", "本事件类别", "配对事件类型", "n", "本事件标记反转%（vpc）"]
    L += ["### T5d 标记逐点相同的配对里，哪一方是反的", "",
          "两队的帧标记相同，必有一方是反的。下表按本事件类别和配对事件类型列出本事件被 vpc 规则判为反转的比例，接近 100 是从属方，接近 0 是主方。",
          "", md(q), ""]


    # 标记反转的帧里 actor 是谁
    rows = []
    for (era, cls), g in u[u["cls"].isin(sec) & (u["vpc_rule"] == "rev")].groupby(["era", "cls"], observed=True):
        if len(g) < 200:
            continue
        rows.append({"时期": era, "事件类别": cls, "vpc 判为反转的帧数": len(g),
                     "actor 的 teammate 为真%": pct(g["act_T"].astype(bool)),
                     "actor 与对方配对帧的 actor 是同一点%": pct(g["pair_actor_same"].astype(float).fillna(0) > 0.5),
                     "整帧取反（含 actor）后队友点数等于厂商本方点数%": pct(g["n_pts"] - g["n_T"] == g["vpc_team"]),
                     "取反但保留 actor 为队友后队友点数等于厂商本方点数%": pct(g["n_pts"] - g["n_T"] + 1 == g["vpc_team"])})
    L += ["### T5e 标记反转的帧里，标为 actor 的点是谁", "",
          "反转帧里标为 actor 的点和对方配对帧的 actor 是同一个点，也就是对方那名球员，不是本事件的球员。",
          "所以摆正时 actor 也要一起取反。后两列用厂商自己的 visible_player_counts 验证这一点。",
          "", md(pd.DataFrame(rows).sort_values(["时期", "vpc 判为反转的帧数"], ascending=[True, False])), ""]


# ── T6 ───────────────────────────────────────────────────────────────

SECONDARY_WHEN_SAME_FLAGS = {"Dribbled Past", "Dispossessed", "Foul Won", "Duel Aerial Lost", "Ball Receipt Incomplete", "50/50 Lost"}


def decide_flip(g):
    """标记是否反转的最终判定  vpc 优先，其次门将，再次配对加类别先验。返回 flip 和用的规则"""
    flip = np.zeros(len(g), bool)
    rule = np.full(len(g), "none", dtype=object)
    v = g["vpc_rule"].astype(str).values
    k = g["keeper_rule"].astype(str).values
    p = g["pair_rule"].astype(str).values
    cls = g["cls"].astype(str).values
    pt = g["pair_type"].astype(str).values
    m = np.isin(v, ["ok", "rev"])
    flip[m], rule[m] = v[m] == "rev", "vpc"
    m2 = ~m & np.isin(k, ["ok", "rev"])
    flip[m2], rule[m2] = k[m2] == "rev", "keeper"
    m3 = ~m & ~m2 & np.isin(p, ["same_flags", "opposite_flags"])
    sec = np.isin(cls, list(SECONDARY_WHEN_SAME_FLAGS)) | ((cls == "Dribble Incomplete") & (pt == "Duel"))
    flip[m3], rule[m3] = (p[m3] == "same_flags") & sec[m3], "pair"
    return flip, rule


def auc(y, x):
    ok = ~np.isnan(x)
    if ok.sum() < 50 or len(np.unique(y[ok])) < 2:
        return np.nan
    a = roc_auc_score(y[ok], x[ok])
    return round(max(a, 1 - a), 3)


def t6_leak(d, scan, L):
    u = d[d["usable"] & (d["era"] != "25/26")].copy()
    u["period"] = np.where(u["era"] == "24/25", "24/25 " + u["state2425"].astype(str) + "状态场次", u["era"])
    tasks = {
        "Dribble": (u["type"].astype(str) == "Dribble", lambda g: (g["outcome"].astype(str) == "Complete").astype(int).values),
        "Ball Receipt": (u["type"].astype(str) == "Ball Receipt*", lambda g: (g["outcome"].astype(str) == "").astype(int).values),
        "Duel（现稿定义，含 Aerial Lost）": (u["type"].astype(str) == "Duel", lambda g: g["outcome"].astype(str).isin({"Won", "Success In Play"}).astype(int).values),
        "Duel 只留 Tackle": ((u["type"].astype(str) == "Duel") & (u["subtype"].astype(str) == "Tackle"), lambda g: g["outcome"].astype(str).isin({"Won", "Success In Play"}).astype(int).values),
        "Pass（对照）": (u["type"].astype(str) == "Pass", lambda g: (g["outcome"].astype(str) == "").astype(int).values),
        "Interception（对照）": (u["type"].astype(str) == "Interception", lambda g: g["outcome"].astype(str).isin({"Won", "Success In Play"}).astype(int).values),
    }
    rows_a, rows_b, rows_c, keep = [], [], [], []
    for task, (mask, lab) in tasks.items():
        g = u[mask].copy()
        g["y"] = lab(g)
        flip, rule = decide_flip(g)
        g["flip"], g["flip_rule"] = flip, rule
        mir = g["mir"].values
        # 摆正后的格内人数  镜像决定取 _s 还是 _m，标记反转决定取 T 还是 O
        def pick(prefix):
            Ts, Os, Tm, Om = (g[f"{prefix}{c}"].values.astype(float) for c in ("T_s", "O_s", "T_m", "O_m"))
            T = np.where(mir, Tm, Ts)
            O = np.where(mir, Om, Os)
            return np.where(flip, O, T), np.where(flip, T, O)
        g["cT_fix"], g["cO_fix"] = pick("c")
        g["nT_fix"], g["nO_fix"] = pick("n")
        # 以帧里 actor 的点为参照自算最近异队球员距离。两人互为最近时这个距离与谁是 actor 无关，
        # 所以标记反转的帧（actor 是对方球员）整帧取反后仍取原始标记下 actor 到最近 teammate 为假的点的距离。
        # 第二个量模拟取反时把 actor 保留为队友的做法，此时量到的是对方球员到他自己队友的距离。
        g["dopp_fix"] = g["dO_act"]
        g["dopp_keep_actor"] = np.where(flip, g["dT_act"], g["dO_act"])
        if task in ("Dribble", "Ball Receipt", "Duel（现稿定义，含 Aerial Lost）"):
            keep.append(g[["season_dir", "match_id", "idx", "type", "subtype", "outcome", "period", "y", "mir", "flip", "flip_rule",
                           "vpc_rule", "keeper_rule", "pair_rule", "cT_s", "cO_s", "cT_fix", "cO_fix"]])
        groups = list(g.groupby("period")) + [("24/25 全部场次", g[g["era"] == "24/25"])]
        for era, h in groups:
            y = h["y"].values
            rows_a.append({"任务": task, "时期": era, "n": len(h), "正例率": round(y.mean(), 3),
                           "负例 坐标镜像%": pct(h.loc[h["y"] == 0, "mir"]), "正例 坐标镜像%": pct(h.loc[h["y"] == 1, "mir"]),
                           "负例 标记反转%": pct(h.loc[h["y"] == 0, "flip"]), "正例 标记反转%": pct(h.loc[h["y"] == 1, "flip"]),
                           "标记无法判定%": pct(h["flip_rule"] == "none")})
            rows_b.append({"任务": task, "时期": era,
                           "球所在格队友数 原始": auc(y, h["cT_s"].values.astype(float)),
                           "球所在格队友数 摆正后": auc(y, h["cT_fix"].values),
                           "球所在格对手数 原始": auc(y, h["cO_s"].values.astype(float)),
                           "球所在格对手数 摆正后": auc(y, h["cO_fix"].values),
                           "3x3 队友数 原始": auc(y, h["nT_s"].values.astype(float)),
                           "3x3 队友数 摆正后": auc(y, h["nT_fix"].values),
                           "3x3 对手数 原始": auc(y, h["nO_s"].values.astype(float)),
                           "3x3 对手数 摆正后": auc(y, h["nO_fix"].values),
                           "厂商标量 最近防守人距离": auc(y, h["sb_dnd"].values.astype(float)),
                           "自算 actor 到最近异队球员距离 整帧取反": auc(y, h["dopp_fix"].values.astype(float)),
                           "同上 取反时保留 actor 为队友": auc(y, h["dopp_keep_actor"].values.astype(float)),
                           "厂商标量 球门侧防守人数": auc(y, h["sb_ngs"].values.astype(float)),
                           "actor 恰在事件位置": auc(y, (h["d_s"].values < 0.01).astype(float))})
            for yy, name in ((1, "正例"), (0, "负例")):
                k = h[h["y"] == yy]
                rows_c.append({"任务": task, "时期": era, "标签": name, "n": len(k),
                               "事件周围 3x3 人数 原始": round((k["nT_s"] + k["nO_s"]).mean(), 2),
                               "镜像位置 3x3 人数 原始": round((k["nT_m"] + k["nO_m"]).mean(), 2),
                               "事件周围 3x3 人数 摆正后": round((k["nT_fix"] + k["nO_fix"]).mean(), 2),
                               "球所在格队友数 原始": round(k["cT_s"].mean(), 2), "球所在格队友数 摆正后": round(k["cT_fix"].mean(), 2),
                               "球所在格对手数 原始": round(k["cO_s"].mean(), 2), "球所在格对手数 摆正后": round(k["cO_fix"].mean(), 2)})
    L += ["### T6a 各任务按标签分组的坐标镜像和标记反转比例（全量）", "",
          "标记反转的最终判定依次用 vpc 规则、门将规则、配对规则，三条都用不上的记为无法判定并按不反转处理。",
          "24/25 按 T2c 的每场状态拆成两段。",
          "", md(pd.DataFrame(rows_a)), "",
          "### T6b 单特征 AUC，摆正前后", "",
          "AUC 取 max(a, 1-a)。球所在格指事件坐标所在的 10x10 码格，与训练用的 8x12 网格一致。摆正指坐标镜像的整帧翻回、标记反转的整帧取反。",
          "厂商标量是 360 文件里预计算的字段，没法摆正，只列原始值的 AUC。自算的两列以帧里 actor 的点为参照，从摆正后的帧重新算，第二列模拟取反时把 actor 保留为队友的做法（T5e 说明了为什么这样不对）。",
          "最后一列是摆正解决不了的残留线索。24/25 正常状态场次的帧本身合规，可当作没有视角问题时各特征 AUC 的参照水平。",
          "", md(pd.DataFrame(rows_b)), "",
          "### T6c 事件周围人数的均值，摆正前后", "", md(pd.DataFrame(rows_c)), ""]
    pd.concat(keep, ignore_index=True).to_parquet(scan / "corrected_flags_dribble_duel_br.parquet", index=False)


# ── T7 ───────────────────────────────────────────────────────────────

def t7_pressure(d, L):
    """逼抢标签用的是项目自己的定义（后续 2 个 on-ball 事件内 possession_team 变化），函数直接取自 prepare_l1_events"""
    from scripts.cache.prepare_l1_events import compute_pressure_labels
    x = d[d["era"] != "25/26"][["season", "match_id", "idx", "type", "team_id", "poss_team_id", "state2425"]].copy()
    x = x.sort_values(["match_id", "idx"]).reset_index(drop=True)
    x["type_name"] = x["type"].astype(str)
    x["possession_team_id"] = x["poss_team_id"]
    x["grp"] = x["season"].astype(str) + np.where(x["state2425"] != "", " " + x["state2425"].astype(str) + "状态场次", "")
    lab = compute_pressure_labels(x)
    pr = x.loc[lab.index].copy()
    pr["y"] = lab.values
    pr["own"] = pr["team_id"] == pr["possession_team_id"]
    chg = (x["possession_team_id"] != x["possession_team_id"].shift()) & (x["match_id"] == x["match_id"].shift())
    x["chg"] = chg
    rows = []
    for g, h in pr.groupby("grp"):
        e = x[x["grp"] == g]
        nm = e["match_id"].nunique()
        rows.append({"分组": g, "场次": nm, "逼抢数": len(h), "正例率": round(h["y"].mean(), 4),
                     "逼抢方等于 possession_team 的占比": round(h["own"].mean(), 4),
                     "每场事件数": round(len(e) / nm), "每场 possession_team 切换次数": round(e["chg"].sum() / nm, 1)})
    L += ["### T7 逼抢任务的正例率和球权切换频率", "",
          "标签函数是 `scripts/cache/prepare_l1_events.py` 的 `compute_pressure_labels`，只用事件文件里的事件类型、顺序和 possession_team，与定格帧无关。",
          "", md(pd.DataFrame(rows)), ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="audit/raw_scan")
    ap.add_argument("--out", default="audit/raw_360_check_tables.md")
    a = ap.parse_args()
    scan = Path(a.scan)
    inv = pd.read_parquet(scan / "match_inventory.parquet")
    meta = pd.read_parquet(scan / "match_metadata.parquet")
    d = add_derived(load_events(scan))
    print(f"[derived] {len(d):,} events", flush=True)
    L = ["# 360 原始数据核查证据表", "",
         f"由 `scripts/audit/analyze_raw_360.py` 从 `{scan}` 的全量扫描结果生成，生成时间 {pd.Timestamp.now():%Y-%m-%d %H:%M}。",
         "扫描脚本 `scripts/audit/scan_raw_360.py`，原始数据在 Google Drive 的 `My Drive/datos`。", ""]
    inv2 = t1_inventory(d, inv, meta, L)
    print("[T1] done", flush=True)
    t2_by_month(d, meta, inv2, L)
    pm = t2_state_2425(d, meta, scan, L)
    d["state2425"] = d["match_id"].map(pm["state"]).fillna("")
    print("[T2] done", flush=True)
    t3_l1(d, L)
    print("[T3] done", flush=True)
    bad = t4_compliance(d, L)
    print("[T4] done", flush=True)
    t5_mechanism(d, bad, L)
    print("[T5] done", flush=True)
    t6_leak(d, scan, L)
    print("[T6] done", flush=True)
    t7_pressure(d, L)
    print("[T7] done", flush=True)
    Path(a.out).write_text("\n".join(L), encoding="utf8")
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
