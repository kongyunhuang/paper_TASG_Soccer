#!/usr/bin/env python3
"""
留出数据确认，第二步  25/26 的结构检查和干净场次清单（不涉及任何模型）
============================
只用事件表、缓存索引、原始扫描结果和厂商元数据，算下面这些，和 22/23、23/24、24/25（干净 202 场和其余 558 场分开）对照。
  一  每场比赛的代理状态口径。24/25 的“状态”是用原始定格帧按三类从属事件的 teammate 标记判的，25/26 只有西甲 29 场扫过帧，
      其余 117 场在不下载的前提下没法同口径判。代理口径有三条，逐场列出
        1  厂商元数据 last_updated_360 晚于 2025-04-14（核查报告说这之后处理的比赛都是正常状态）
        2  射门（不含点球）带帧率不低于 70%（和 24/25 的 clean_matches 同一条）
        3  出界传球带帧率不低于 50%（24/25 那 27 场“状态正常但帧随结果”的比赛出界传球带帧率只有 6% 到 7%，干净场次 88%）
      29 场实扫的，另给出按 24/25 同一规则算的反转比例和状态。
  二  各任务的带帧比例按赛季对照；传球按结果（完成、未完成、出界、其他）、射门按结果分别算带帧率。
  三  Pressure 的 5 秒标签在含 25/26 的事件表上重算，各赛季正例率；其余四个任务的正例率按赛季对照。
  四  四个厂商 360 标量的均值和标准差按赛季对照（输入分布，不是模型成绩）。
  五  缓存索引的覆盖是否等于事件表的 has_360（逐任务逐赛季核对），落点任务可用条数。
“同一个键出现两次就报错”的检查加在所有分组表上。

输入  data/cache/L1_events_v3.parquet，data/cache/action_soccermaps_*_idx.parquet，audit/raw_scan/{match_metadata,match_inventory,match_state_2425}.parquet，
      audit/raw_scan/events/11_318_2025_26_part.parquet
输出  audit/holdout_2526_structure_check.md
      data/cache/holdout_2526/match_table_2526.parquet     146 场逐场的代理口径和各项带帧率
      data/cache/holdout_2526/clean_matches_2526.parquet   通过代理口径的场次清单

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/eval/holdout_2526_structure_check.py

Last modified 2026-10-02
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.eval.summarize_u1_clean import clean_matches
from scripts.training.train_unified_clean import F_SB, pressure_labels_5s

CACHE = Path("data/cache")
RAW = Path("audit/raw_scan")
OUT_DIR = CACHE / "holdout_2526"
OUT_MD = Path("audit/holdout_2526_structure_check.md")
SEASON_ORDER = ["22/23", "23/24", "24/25 干净 202", "24/25 其余 558", "25/26"]
TASK_TYPES = {"pass": "Pass", "shot": "Shot", "interception": "Interception", "ball_recovery": "Ball Recovery",
              "pressure": "Pressure", "dribble": "Dribble", "duel": "Duel"}
STATE_DATE = pd.Timestamp("2025-04-14")
SHOT_MIN, OUT_MIN = 70.0, 50.0


def md(df, index=False):
    return df.to_markdown(index=index)


def no_dup(keys, what):
    s = pd.Series(list(keys))
    d = s[s.duplicated()]
    if len(d):
        raise RuntimeError(f"{what}  同一个键出现两次  {d.unique()[:5]}")


def season_group(df, clean2425):
    s = df["season_dir"].astype(str)
    g = np.where(s.str.contains("2022_23"), "22/23", np.where(s.str.contains("2023_24"), "23/24", np.where(s.str.contains("2025_26"), "25/26", "")))
    g = pd.Series(g, index=df.index)
    m2425 = s.str.contains("2024_25")
    g[m2425 & df["match_id"].astype(int).isin(clean2425)] = "24/25 干净 202"
    g[m2425 & ~df["match_id"].astype(int).isin(clean2425)] = "24/25 其余 558"
    return pd.Categorical(g, categories=SEASON_ORDER, ordered=True)


def pct(x):
    return round(100 * float(np.mean(x)), 1) if len(x) else float("nan")


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["event_id", "match_id", "period", "timestamp", "team_id", "type_name", "season_dir", "has_360",
            "pass_outcome_name", "shot_outcome_name", "shot_type_name", "interception_outcome_name", "ball_recovery_failure",
            "pass_end_location_x", "pass_end_location_y", "dribble_outcome_name", "duel_type_name", "duel_outcome_name"] + F_SB
    df = pd.read_parquet(CACHE / "L1_events_v3.parquet", columns=cols)
    clean2425 = clean_matches()
    df["grp"] = season_group(df, clean2425)
    df["match_id"] = df["match_id"].astype(int)
    print(f"[load] {len(df):,} events, groups {df['grp'].value_counts().to_dict()}  ({time.time() - t0:.0f}s)", flush=True)

    # 缓存索引  逐任务的 event_id 集合
    idx = {t: set(pq.read_table(CACHE / f"action_soccermaps_{t}_idx.parquet", columns=["event_id"]).column(0).to_pylist()) for t in TASK_TYPES}
    for t in TASK_TYPES:
        df[f"in_{t}"] = df["event_id"].isin(idx[t])

    L = ["# 留出数据确认，25/26 的结构检查（不涉及任何模型）", "",
         f"由 `scripts/eval/holdout_2526_structure_check.py` 生成，{time.strftime('%Y-%m-%d %H:%M')}。24/25 的干净 202 场取 `summarize_u1_clean.clean_matches()`。", ""]

    # 五  缓存覆盖对 has_360
    rows = []
    for t, tn in TASK_TYPES.items():
        d = df[df["type_name"] == tn]
        for g, h in d.groupby("grp", observed=True):
            rows.append({"任务": t, "赛季": g, "事件数": len(h), "has_360": int(h["has_360"].sum()), "在缓存索引": int(h[f"in_{t}"].sum()),
                         "has_360 但不在缓存": int((h["has_360"] & ~h[f"in_{t}"]).sum()), "在缓存但无 has_360": int((~h["has_360"].astype(bool) & h[f"in_{t}"]).sum())})
    cov = pd.DataFrame(rows)
    no_dup(zip(cov["任务"], cov["赛季"]), "缓存覆盖表")
    L += ["## 五  缓存索引对事件表 has_360 的覆盖（逐任务逐赛季）", "", md(cov), ""]
    print("[cov] has_360 但不在缓存 合计", int(cov["has_360 但不在缓存"].sum()), "；在缓存但无 has_360 合计", int(cov["在缓存但无 has_360"].sum()), flush=True)

    # 二  带帧比例按赛季，按结果
    rows = []
    for t, tn in TASK_TYPES.items():
        d = df[df["type_name"] == tn]
        if t == "shot":
            d = d[d["shot_type_name"] != "Penalty"]
        for g, h in d.groupby("grp", observed=True):
            rows.append({"任务": t + ("（不含点球）" if t == "shot" else ""), "赛季": g, "事件数": len(h), "带帧率%": pct(h[f"in_{t}"])})
    fr = pd.DataFrame(rows)
    no_dup(zip(fr["任务"], fr["赛季"]), "带帧率表")
    L += ["## 二  各任务带帧率（按缓存索引）按赛季对照", "", md(fr.pivot(index="任务", columns="赛季", values="带帧率%").reset_index()), "",
          md(fr.pivot(index="任务", columns="赛季", values="事件数").reset_index()), ""]

    p = df[df["type_name"] == "Pass"].copy()
    p["结果"] = p["pass_outcome_name"].fillna("完成").replace({"Incomplete": "未完成", "Out": "出界", "Pass Offside": "越位", "Unknown": "未知", "Injury Clearance": "伤停解围"})
    rows = []
    for (g, o), h in p.groupby(["grp", "结果"], observed=True):
        rows.append({"赛季": g, "结果": o, "事件数": len(h), "带帧率%": pct(h["in_pass"])})
    po = pd.DataFrame(rows)
    no_dup(zip(po["赛季"], po["结果"]), "传球结果带帧表")
    L += ["### 传球按结果的带帧率", "", md(po.pivot(index="结果", columns="赛季", values="带帧率%").reset_index()), "",
          md(po.pivot(index="结果", columns="赛季", values="事件数").reset_index()), ""]

    s = df[(df["type_name"] == "Shot") & (df["shot_type_name"] != "Penalty")].copy()
    rows = []
    for (g, o), h in s.groupby(["grp", "shot_outcome_name"], observed=True):
        rows.append({"赛季": g, "结果": o, "事件数": len(h), "带帧率%": pct(h["in_shot"])})
    so = pd.DataFrame(rows)
    no_dup(zip(so["赛季"], so["结果"]), "射门结果带帧表")
    L += ["### 射门（不含点球）按结果的带帧率", "", md(so.pivot(index="结果", columns="赛季", values="带帧率%").reset_index()), "",
          md(so.pivot(index="结果", columns="赛季", values="事件数").reset_index()), ""]

    # 三  标签正例率
    press_y = pressure_labels_5s(df)
    df["press_y"] = press_y.reindex(df["event_id"].values).values
    rows = []
    for g, h in df.groupby("grp", observed=True):
        pr = h[(h["type_name"] == "Pressure") & h["in_pressure"]]
        pa = h[(h["type_name"] == "Pass") & h["in_pass"]]
        sh = h[(h["type_name"] == "Shot") & (h["shot_type_name"] != "Penalty") & h["in_shot"]]
        ic = h[(h["type_name"] == "Interception") & h["in_interception"]]
        br = h[(h["type_name"] == "Ball Recovery") & h["in_ball_recovery"]]
        dr = h[(h["type_name"] == "Dribble") & h["in_dribble"]]
        tk = h[(h["type_name"] == "Duel") & (h["duel_type_name"] == "Tackle") & h["duel_outcome_name"].isin(["Won", "Success", "Success In Play", "Success Out", "Lost In Play", "Lost Out"]) & h["in_duel"]]
        rows.append({"赛季": g, "Pressure 5 秒正例率%": pct(pr["press_y"].fillna(0)), "Pressure 带帧条数": len(pr),
                     "Pass 完成率%": pct(pa["pass_outcome_name"].isna()), "Shot 进球率%": pct(sh["shot_outcome_name"] == "Goal"),
                     "Interception 成功率%": pct(ic["interception_outcome_name"].isin(["Won", "Success In Play"])),
                     "Ball Recovery 成功率%": pct(br["ball_recovery_failure"] != True),
                     "Dribble 成功率%": pct(dr["dribble_outcome_name"] == "Complete"),
                     "Tackle 成功率%": pct(tk["duel_outcome_name"].isin(["Won", "Success", "Success In Play", "Success Out"])), "Tackle 带帧有标签条数": len(tk),
                     "落点可用条数（带帧且有终点）": int((pa["pass_end_location_x"].notna() & pa["pass_end_location_y"].notna()).sum())})
    lab = pd.DataFrame(rows)
    no_dup(lab["赛季"], "标签表")
    L += ["## 三  各任务标签正例率按赛季（只取带帧事件，和训练口径相同）", "", md(lab), "",
          "Pressure 的 5 秒标签用 `train_unified_clean.pressure_labels_5s` 在含 25/26 的整张事件表上重算。Dribble 和 Tackle 这里用的是 4 月旧缓存的带帧标记，只看标签分布，张量本身不能用。", ""]

    # 四  标量分布
    rows = []
    for g, h in df[df["has_360"].astype(bool)].groupby("grp", observed=True):
        r = {"赛季": g, "带帧事件数": len(h)}
        for c in F_SB:
            r[c + " 均值"] = round(float(h[c].mean()), 3)
            r[c + " 标准差"] = round(float(h[c].std()), 3)
            r[c + " 缺失%"] = pct(h[c].isna())
        rows.append(r)
    sc = pd.DataFrame(rows)
    no_dup(sc["赛季"], "标量表")
    L += ["## 四  厂商 360 标量的分布按赛季（全部带帧事件）", "", md(sc), ""]

    # 一  25/26 逐场
    meta = pd.read_parquet(RAW / "match_metadata.parquet")
    meta = meta[meta["season_dir"].str.contains("2025_26") & (meta["match_status_360"] == "available")].copy()
    meta["match_id"] = meta["match_id"].astype(int)
    no_dup(meta["match_id"], "25/26 元数据")
    inv = pd.read_parquet(RAW / "match_inventory.parquet")
    inv = inv[inv["season_dir"].str.contains("2025_26")].copy()
    inv["match_id"] = inv["match_id"].astype(int)
    no_dup(inv["match_id"], "25/26 库存")
    h = df[df["grp"] == "25/26"]
    rows = []
    for mid, g in h.groupby("match_id"):
        sh = g[(g["type_name"] == "Shot") & (g["shot_type_name"] != "Penalty")]
        pa = g[g["type_name"] == "Pass"]
        out = pa[pa["pass_outcome_name"] == "Out"]
        rows.append({"match_id": mid, "season_dir": g["season_dir"].iloc[0], "n_events": len(g), "frame_pct": pct(g["has_360"].astype(bool)),
                     "shots": len(sh), "shot_frame_pct": pct(sh["in_shot"]), "goal_frame_pct": pct(sh.loc[sh["shot_outcome_name"] == "Goal", "in_shot"]),
                     "nongoal_frame_pct": pct(sh.loc[sh["shot_outcome_name"] != "Goal", "in_shot"]),
                     "passes": len(pa), "pass_frame_pct": pct(pa["in_pass"]), "out_passes": len(out), "out_pass_frame_pct": pct(out["in_pass"]),
                     "incomplete_pass_frame_pct": pct(pa.loc[pa["pass_outcome_name"] == "Incomplete", "in_pass"]),
                     "complete_pass_frame_pct": pct(pa.loc[pa["pass_outcome_name"].isna(), "in_pass"]),
                     "pressures": int(((g["type_name"] == "Pressure") & g["in_pressure"]).sum()),
                     "interceptions": int(((g["type_name"] == "Interception") & g["in_interception"]).sum()),
                     "ball_recoveries": int(((g["type_name"] == "Ball Recovery") & g["in_ball_recovery"]).sum())})
    mt = pd.DataFrame(rows)
    no_dup(mt["match_id"], "25/26 逐场表")
    mt = mt.merge(meta[["match_id", "match_date", "last_updated", "last_updated_360"]], on="match_id", how="left")
    mt = mt.merge(inv[["match_id", "events_local", "f360_local", "f360_bytes", "events_bytes"]], on="match_id", how="left")
    assert mt["match_date"].notna().all(), "有比赛在元数据里找不到"
    mt["lu360"] = pd.to_datetime(mt["last_updated_360"])
    mt["crit1_lu360_after"] = mt["lu360"] > STATE_DATE
    mt["crit2_shot_frame"] = mt["shot_frame_pct"] >= SHOT_MIN
    mt["crit3_out_pass_frame"] = mt["out_pass_frame_pct"] >= OUT_MIN

    # 29 场实扫的状态，和 24/25 同一规则
    sc_path = RAW / "events" / "11_318_2025_26_part.parquet"
    raw = pd.read_parquet(sc_path, columns=["match_id", "idx", "type", "subtype", "outcome", "has_frame", "loc_x", "n_act", "n_T", "vpc_team", "vpc_other", "act_x", "act_y", "loc_y"])
    raw["match_id"] = raw["match_id"].astype(int)
    scanned = raw[raw["has_frame"]]["match_id"].unique()
    ok = raw["has_frame"] & raw["loc_x"].notna() & (raw["n_act"] > 0)
    has = ok & (raw["vpc_team"] >= 0) & (raw["vpc_other"] >= 0)
    tie = has & (raw["vpc_team"] == raw["vpc_other"])
    f = np.full(len(raw), "na", dtype=object)
    f[(has & ~tie & (raw["n_T"] == raw["vpc_team"])).values] = "ok"
    f[(has & ~tie & (raw["n_T"] == raw["vpc_other"])).values] = "rev"
    raw["vpc_rule"] = f
    d_s = np.hypot(raw["act_x"] - raw["loc_x"], raw["act_y"] - raw["loc_y"])
    d_m = np.hypot(raw["act_x"] - (120 - raw["loc_x"]), raw["act_y"] - (80 - raw["loc_y"]))
    raw["mir"] = ok & (d_m < d_s)
    t_, sub, out_ = raw["type"].astype(str), raw["subtype"].astype(str), raw["outcome"].astype(str)
    raw["cls"] = np.where(t_ == "Dribble", "Dribble " + out_, np.where(t_ == "Duel", "Duel " + sub, t_))
    v = raw[ok & raw["cls"].isin(["Dispossessed", "Foul Won", "Dribbled Past"]) & raw["vpc_rule"].isin(["ok", "rev"])]
    st = v.groupby("match_id").agg(rev_pct=("vpc_rule", lambda s: 100 * (s == "rev").mean()), n_state_events=("idx", "size"))

    def rev_of(c):
        x = raw[ok & (raw["cls"] == c) & raw["vpc_rule"].isin(["ok", "rev"])]
        return x.groupby("match_id")["vpc_rule"].agg(lambda s: 100 * (s == "rev").mean())
    st["dribble_incomplete_rev_pct"] = rev_of("Dribble Incomplete")
    st["aerial_lost_rev_pct"] = rev_of("Duel Aerial Lost")
    st["mirror_pct_all"] = raw[ok].groupby("match_id")["mir"].mean() * 100
    st["scan_state"] = np.where(st["rev_pct"] >= 80, "反转", np.where(st["rev_pct"] <= 20, "正常", "中间"))
    st = st.reset_index()
    no_dup(st["match_id"], "29 场状态表")
    mt = mt.merge(st, on="match_id", how="left")
    assert len(scanned) == st["match_id"].nunique(), f"扫过帧的场次 {len(scanned)} 和状态表 {st['match_id'].nunique()} 不等"
    mt["scan_state"] = mt["scan_state"].fillna("未扫帧")
    mt["clean_proxy"] = mt["crit1_lu360_after"] & mt["crit2_shot_frame"] & mt["crit3_out_pass_frame"] & (mt["scan_state"] != "反转") & (mt["scan_state"] != "中间")
    mt = mt.sort_values(["season_dir", "match_date", "match_id"]).reset_index(drop=True)
    mt.to_parquet(OUT_DIR / "match_table_2526.parquet", index=False)
    clean = mt[mt["clean_proxy"]]
    clean[["match_id", "season_dir", "match_date"]].to_parquet(OUT_DIR / "clean_matches_2526.parquet", index=False)

    L += ["## 一  25/26 逐场的代理口径", "",
          f"146 场里，last_updated_360 晚于 {STATE_DATE:%Y-%m-%d} 的 {int(mt['crit1_lu360_after'].sum())} 场（最早 {mt['lu360'].min():%Y-%m-%d}，最晚 {mt['lu360'].max():%Y-%m-%d}）；"
          f"射门带帧率不低于 {SHOT_MIN:.0f}% 的 {int(mt['crit2_shot_frame'].sum())} 场（最低 {mt['shot_frame_pct'].min():.1f}%）；"
          f"出界传球带帧率不低于 {OUT_MIN:.0f}% 的 {int(mt['crit3_out_pass_frame'].sum())} 场（最低 {mt['out_pass_frame_pct'].min():.1f}%，中位 {mt['out_pass_frame_pct'].median():.1f}%）。",
          f"西甲 {len(st)} 场实扫过帧，按 24/25 同一规则判，状态分布 {st['scan_state'].value_counts().to_dict()}，三类从属事件反转比例最大 {st['rev_pct'].max():.1f}%，"
          f"Dribble Incomplete 反转比例场均 {st['dribble_incomplete_rev_pct'].mean():.1f}%，Aerial Lost 反转比例场均 {st['aerial_lost_rev_pct'].mean():.1f}%，全部带帧事件的坐标镜像比例场均 {st['mirror_pct_all'].mean():.2f}%。",
          f"三条代理口径加实扫状态都通过的 **{len(clean)} 场**（西甲 {int((clean['season_dir'].str.startswith('11_')).sum())}，英超 {int((clean['season_dir'].str.startswith('2_')).sum())}），清单在 `data/cache/holdout_2526/clean_matches_2526.parquet`。", "",
          "逐场（按联赛和日期排）", "",
          md(mt[["match_id", "season_dir", "match_date", "last_updated_360", "n_events", "frame_pct", "shots", "shot_frame_pct", "goal_frame_pct", "nongoal_frame_pct",
                 "out_passes", "out_pass_frame_pct", "incomplete_pass_frame_pct", "complete_pass_frame_pct", "pressures", "interceptions", "ball_recoveries",
                 "scan_state", "rev_pct", "dribble_incomplete_rev_pct", "aerial_lost_rev_pct", "events_local", "f360_local", "clean_proxy"]].round(1)), ""]
    # 24/25 参照  27 场被 shot_frame_pct 剔掉的“正常”比赛的出界传球带帧率，证明第三条口径的依据
    st2425 = pd.read_parquet(RAW / "match_state_2425.parquet")
    st2425["match_id"] = st2425["match_id"].astype(int)
    p2425 = df[df["season_dir"].str.contains("2024_25") & (df["type_name"] == "Pass") & (df["pass_outcome_name"] == "Out")]
    opm = p2425.groupby("match_id")["in_pass"].mean().mul(100).rename("out_pass_frame_pct").reset_index()
    opm = opm.merge(st2425[["match_id", "state", "shot_frame_pct"]], on="match_id", how="left")
    opm["组"] = np.where(opm["state"] != "正常", "反转或中间", np.where(opm["shot_frame_pct"] >= 70, "干净 202", "正常但射门带帧低 27"))
    ref = opm.groupby("组")["out_pass_frame_pct"].agg(["count", "min", "median", "max"]).round(1).reset_index()
    L += ["24/25 参照，出界传球带帧率按场次组（第三条口径的依据）", "", md(ref), ""]
    OUT_MD.write_text("\n".join(L), encoding="utf8")
    print(f"[clean] 25/26 通过代理口径 {len(clean)} / {len(mt)} 场", flush=True)
    print(f"[done] {OUT_MD}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
