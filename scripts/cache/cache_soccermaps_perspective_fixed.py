#!/usr/bin/env python3
"""
视角摆正后的空间张量  给 Dribble 和 Duel 重建 SoccerMap 和 360 标量
============================
StatsBomb 360 的定格帧在失败的过人和输掉的空中对抗上是从对方视角给的（见 audit/frame_perspective_report.md）。
本脚本把这两类事件的每一帧摆回事件所属球队的坐标方向，摆正规则不使用结果标签。
  坐标  帧里标为 actor 的球员若离镜像位置 (120-x, 80-y) 比离事件位置更近，整帧做镜像
  平移  第四版起，镜像之后再把整帧平移到 actor 落在事件坐标上（平移量存在 actor_offset 列）
  标记  第三版起不再翻转 teammate 标记。下面三条规则只算出来存成诊断列（keeper_swap、twin_swap、prev_swap、flag_swapped），不改数据
        24/25 有 531 场处在标记反转的状态，那些场次的失败帧这里修不了，评估时按场次状态排除
        keeper  看得到靠近球门的门将时，本方门将应在 x 小于 60 一侧
        twin    相邻事件里有与本帧点集相同的帧时，按那一帧的标记和它所属球队推出本帧应有的标记（只用于 22/23 和 23/24）
        prev    否则和本方上一个事件的帧比位置连续性
同时从摆正后的帧重算四个 360 标量，不再使用厂商预计算的 sb_ 字段。
参照点一律用帧里 actor 的位置，不用事件坐标。原因是帧的归属事件的 actor 会被放在该事件的精确坐标上，
失败过人的帧归属于对方的铲球事件，若用过人事件坐标当参照点，铲球人到参照点的距离恒为 0.14 码，等于把结果写进了特征
（2026-10-02 第一版就犯了这个错，B2 验证集 AUC 0.95）。标记翻转时 actor 自己始终保持为本方。

输入  原始 StatsBomb 文件 *_events.json 和 *_360.json（Google Drive 的 datos 目录，6 个完整赛季）
输出  data/cache/perspective_fixed/soccermaps_{dribble,duel}.npy       (N, 7, 8, 12) float32
      data/cache/perspective_fixed/soccermaps_{dribble,duel}_idx.parquet  event_id、重算的标量、每条规则的判断
      audit/perspective_fix_validation.md                              摆正前后的校验表

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/cache/cache_soccermaps_perspective_fixed.py
  PYTHONPATH=. python -u scripts/cache/cache_soccermaps_perspective_fixed.py --limit 20   （每赛季只跑 20 场，试跑用）

Last modified 2026-10-02
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cache.cache_action_soccermaps import DATA_DIR, create_soccermap

SEASONS = ["2_235_2022_23", "2_281_2023_24", "2_317_2024_25", "11_235_2022_23", "11_281_2023_24", "11_317_2024_25"]
OUT_DIR = Path("data/cache/perspective_fixed")
TARGETS = {"Dribble": "dribble", "Duel": "duel"}
# 这些类型的帧自身可能有视角问题，不拿来当参照
UNTRUSTED = {"Dribble", "Ball Receipt*"}
MIRROR = np.array([120.0, 80.0])


def frame_arrays(ff):
    P = np.array([[p["location"][0], p["location"][1]] for p in ff], dtype=float).reshape(-1, 2)
    mate = np.array([bool(p.get("teammate")) for p in ff])
    keeper = np.array([bool(p.get("keeper")) for p in ff])
    actor = np.array([bool(p.get("actor")) for p in ff])
    return P, mate, keeper, actor


def actor_mirrored(P, actor, loc):
    """actor 离镜像位置更近返回 True，没有 actor 返回 None"""
    if not actor.any():
        return None
    a = P[actor][0]
    return bool(((a - (MIRROR - loc)) ** 2).sum() < ((a - loc) ** 2).sum())


def keeper_rule(P, mate, keeper):
    """返回 True 表示标记反了，False 表示正常，None 表示没有可用的门将"""
    k = keeper & ((P[:, 0] < 30) | (P[:, 0] > 90))
    if not k.any():
        return None
    return bool(np.mean((P[k, 0] < 60) != mate[k]) > 0.5)


def same_points(A, B, tol=0.6):
    if len(A) != len(B) or len(A) == 0:
        return None
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    if (d.min(1) < tol).all() and (d.min(0) < tol).all():
        return d.argmin(1)
    return None


def continuity_cost(P, mate, R, rmate, cap=12.0):
    c = []
    for f in (True, False):
        A, B = P[mate == f], R[rmate == f]
        if len(A) == 0:
            continue
        if len(B) == 0:
            c += [cap] * len(A)
            continue
        c += list(np.minimum(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1), cap))
    return float(np.mean(c)) if c else np.nan


def is_aerial_lost(e):
    return e["type"]["name"] == "Duel" and not (e.get("duel", {}) or {}).get("outcome")


def trusted(e):
    return e["type"]["name"] not in UNTRUSTED and not is_aerial_lost(e)


def process_match(args):
    season, fe, match_date = args
    f360 = fe.replace("_events.json", "_360.json")
    if not Path(f360).exists():
        return [], None
    ev = json.load(open(fe, encoding="utf-8"))
    fr = {x["event_uuid"]: x for x in json.load(open(f360, encoding="utf-8"))}
    cache = {}

    def get(i):
        if i not in cache:
            e = ev[i]
            ff = (fr.get(e["id"]) or {}).get("freeze_frame")
            cache[i] = frame_arrays(ff) if ff else None
        return cache[i]

    rows, maps = [], []
    for i, e in enumerate(ev):
        t = e["type"]["name"]
        if t not in TARGETS or "location" not in e:
            continue
        fa = get(i)
        if fa is None:
            continue
        P, mate, keeper, actor = fa
        loc = np.array(e["location"][:2], dtype=float)
        mir = actor_mirrored(P, actor, loc)
        if mir is None:
            continue
        if mir:
            P = MIRROR - P
        team = e["team"]["id"]

        k_swap = keeper_rule(P, mate, keeper)

        t_swap, twin_type = None, ""
        order = [j for d in range(1, 9) for j in (i - d, i + d) if 0 <= j < len(ev)]
        for j in order:
            if not trusted(ev[j]):
                continue
            fb = get(j)
            if fb is None:
                continue
            B, bmate = fb[0], fb[1]
            m = same_points(P, B)
            if m is None:
                m = same_points(P, MIRROR - B)
            if m is None:
                continue
            eq = float(np.mean(mate == bmate[m]))
            same_team = ev[j]["team"]["id"] == team
            t_swap = bool((eq > 0.5) != same_team)
            twin_type = ev[j]["type"]["name"] + ("(own)" if same_team else "(opp)")
            break

        p_swap = None
        for j in range(i - 1, max(i - 8, -1), -1):
            p = ev[j]
            if p["team"]["id"] != team or not trusted(p) or "location" not in p:
                continue
            fb = get(j)
            if fb is None:
                continue
            R, rmate, _, ractor = fb
            if actor_mirrored(R, ractor, np.array(p["location"][:2], dtype=float)) is not False:
                continue
            ck, cs = continuity_cost(P, mate, R, rmate), continuity_cost(P, ~mate, R, rmate)
            if np.isfinite(ck) and np.isfinite(cs):
                p_swap = bool(cs < ck)
            break

        # 24/25 的新管线里，共用快照的两个事件标记不再互为相反，twin 规则在那里不可靠
        # （试跑时它和门将规则在铲球事件上几乎全不一致），所以只在旧管线的赛季用
        new_pipeline = "2024_25" in season
        if k_swap is not None:
            swap, rule = k_swap, "keeper"
        elif t_swap is not None and not new_pipeline:
            swap, rule = t_swap, "twin"
        elif p_swap is not None:
            swap, rule = p_swap, "prev"
        else:
            swap, rule = False, "none"
        # 第三版（2026-10-02）  不再翻转队友标记，三条规则的判断只作为诊断列保留。
        # 依据是核查会话 xcv2-ae 的全量核对（audit/raw_360_check_tables.md）。
        # 22/23、23/24 和 24/25 的正常状态场次里标记本来就对，翻转只会带进误判（第二版在正常状态场次误翻了 5.3%）。
        # 24/25 的反转状态场次（531 场）里，失败事件的帧整帧来自对方的成对事件，连 actor 都是对方球员，
        # 这里不处理，这些场次不进入干净测试集，逐场状态见 audit/raw_scan/match_state_2425.parquet。
        # 第四版（2026-10-02）  整帧平移，使 actor 落在事件坐标上。
        # 帧的主方事件的球员被放在事件的精确坐标上，其余是观测位置。旧赛季失败过人的帧主方是铲球事件，
        # 过人者在约 1 码外的观测位置上，成功过人的帧里过人者恰在事件坐标上。“actor 是否恰在事件坐标上”
        # 在旧赛季单特征 AUC 0.943（核查会话 xcv2-ae 的 T 表），事件特征里有精确坐标，CNN 的到球距离通道又能反推
        # actor 位置，两者一比就是结果。平移之后所有帧都是 actor 在事件坐标上、对手在相对位置上，结构一致。
        actor_offset = float(np.sqrt(((P[actor][0] - loc) ** 2).sum()))
        P = P + (loc - P[actor][0])
        ref = loc

        opp = P[~mate]
        mates_other = P[mate & ~actor]
        d_def = float(np.sqrt(((opp - ref) ** 2).sum(1)).min()) if len(opp) else np.nan
        if t == "Dribble":
            out = e["dribble"]["outcome"]["name"]
            sub = ""
        else:
            out = ((e.get("duel", {}) or {}).get("outcome") or {}).get("name", "")
            sub = ((e.get("duel", {}) or {}).get("type") or {}).get("name", "")
        rows.append(dict(event_id=e["id"], match_id=int(Path(fe).stem.replace("_events", "")), season_dir=season,
                         match_date=match_date, actor_offset=actor_offset,
                         type_name=t, outcome=out, duel_type=sub,
                         mirrored=mir, flag_swapped=swap, rule=rule,
                         keeper_swap=k_swap, twin_swap=t_swap, prev_swap=p_swap, twin_type=twin_type,
                         fx_distance_to_nearest_defender=d_def,
                         fx_num_defenders_on_goal_side=int((opp[:, 0] > ref[0]).sum()) if len(opp) else 0,
                         fx_visible_teammates=int(len(mates_other)), fx_visible_opponents=int(len(opp))))
        maps.append(create_soccermap(ref[0], ref[1], P[mate].tolist(), P[~mate].tolist()))
    return rows, (np.stack(maps) if maps else None)


def pct(x):
    x = x.dropna()
    return round(100 * x.astype(bool).mean(), 1) if len(x) else np.nan


def single_feature_auc(df):
    """按三个时期看重算的标量单独能把结果分到什么程度。2025-04 之后的比赛原始帧基本干净，当作参照。"""
    from sklearn.metrics import roc_auc_score
    df = df.copy()
    state_path = Path("audit/raw_scan/match_state_2425.parquet")
    if state_path.exists():
        st = pd.read_parquet(state_path)
        normal = set(st.loc[st["state"] == "正常", "match_id"].astype(int))
        is_normal = df["match_id"].astype(int).isin(normal)
        df["period"] = np.where(~df["season_dir"].str.contains("2024_25"), "A 22/23 and 23/24",
                                np.where(is_normal, "C 24/25 normal-state matches", "B 24/25 reversed-state matches"))
    else:
        df["period"] = np.where(~df["season_dir"].str.contains("2024_25"), "A 22/23 and 23/24",
                                np.where(df["match_date"] >= "2025-04-01", "C 24/25 Apr-May", "B 24/25 Aug-Mar"))
    df["fx_actor_offset_gt_half_yard"] = (df["actor_offset"] > 0.5).astype(int)
    rows = []
    tasks = {"Dribble": (df["type_name"] == "Dribble", lambda d: (d["outcome"] == "Complete").astype(int)),
             "Tackle": ((df["type_name"] == "Duel") & (df["duel_type"] == "Tackle") & (df["outcome"] != ""),
                        lambda d: d["outcome"].isin({"Won", "Success", "Success In Play", "Success Out"}).astype(int))}
    for task, (mask, lab) in tasks.items():
        d = df[mask]
        for period, g in d.groupby("period"):
            y = lab(g)
            r = dict(task=task, period=period, n=len(g), pos_rate=round(y.mean(), 3))
            for c in [c for c in g.columns if c.startswith("fx_")]:
                a = roc_auc_score(y, g[c].fillna(-1))
                r[c.replace("fx_", "auc_")] = round(max(a, 1 - a), 3)
            rows.append(r)
    return pd.DataFrame(rows)


def validation_report(df, path):
    auc_table = single_feature_auc(df)
    df = df.copy()
    df["era"] = np.where(df["season_dir"].str.contains("2024_25"), "24/25", "22/23 and 23/24")
    df["group"] = np.where(df["type_name"] == "Dribble", "Dribble " + df["outcome"],
                           np.where(df["outcome"] == "", "Duel aerial lost", "Duel tackle " + df["outcome"]))
    rows = []
    for (era, g), d in df.groupby(["era", "group"]):
        kn = d[d["keeper_swap"].notna()]
        both_t = kn[kn["twin_swap"].notna()]
        both_p = kn[kn["prev_swap"].notna()]
        rows.append(dict(seasons=era, group=g, n=len(d), mirrored_pct=pct(d["mirrored"]),
                         swapped_pct=pct(d["flag_swapped"]),
                         rule_keeper_pct=round(100 * (d["rule"] == "keeper").mean(), 1),
                         rule_twin_pct=round(100 * (d["rule"] == "twin").mean(), 1),
                         rule_prev_pct=round(100 * (d["rule"] == "prev").mean(), 1),
                         rule_none_pct=round(100 * (d["rule"] == "none").mean(), 1),
                         keeper_says_swapped_pct=pct(kn["keeper_swap"]),
                         twin_agrees_with_keeper_pct=round(100 * (both_t["twin_swap"] == both_t["keeper_swap"]).mean(), 1) if len(both_t) else np.nan,
                         prev_agrees_with_keeper_pct=round(100 * (both_p["prev_swap"] == both_p["keeper_swap"]).mean(), 1) if len(both_p) else np.nan))
    lines = ["# 视角摆正校验", "", "由 `scripts/cache/cache_soccermaps_perspective_fixed.py` 生成。", "",
             "mirrored_pct 是按 actor 位置判定为镜像并已摆正的比例。swapped_pct 是最终判定标记反了并已翻回的比例。",
             "rule_ 四列是最终采用哪条规则的占比。keeper_says_swapped_pct 是看得到门将的帧里门将规则判定为反的比例。",
             "后两列是在看得到门将的帧上，另外两条规则和门将规则一致的比例，用来估计没有门将时的判断准确率。", "",
             pd.DataFrame(rows).to_markdown(index=False), "",
             "## 重算的四个标量各自的单特征 AUC", "",
             "三个时期应当接近。C 是原始帧合规的场次，A 若明显高于 C，说明摆正后训练数据里仍有随结果而变的痕迹。B 是标记反转的场次，本脚本不修，预期会偏高。",
             "最后一列 actor_offset_gt_half_yard 是平移前 actor 离事件坐标是否超过半码，它本身就能分出结果，平移就是为了消掉它，这一列不进模型。", "",
             auc_table.to_markdown(index=False), ""]
    Path(path).write_text("\n".join(lines), encoding="utf8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    jobs = []
    for s in SEASONS:
        info = json.load(open(next((DATA_DIR / s).glob("00_info_all_matches_*.json")), encoding="utf-8"))
        dates = {int(m["match_id"]): str(m.get("match_date")) for m in info}
        files = sorted(str(p) for p in (DATA_DIR / s).glob("*_events.json"))
        if a.limit:
            files = files[:a.limit]
        jobs += [(s, f, dates.get(int(Path(f).stem.replace("_events", "")), "")) for f in files]
    print(f"[start] {len(jobs)} matches, workers={a.workers}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_rows, all_maps = [], []
    with Pool(a.workers) as pool:
        for n, (rows, maps) in enumerate(pool.imap(process_match, jobs, chunksize=4), 1):
            if rows:
                all_rows += rows
                all_maps.append(maps)
            if n % 100 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"[{n}/{len(jobs)}] events so far {len(all_rows):,}  elapsed {el / 60:.1f} min  eta {el / n * (len(jobs) - n) / 60:.0f} min", flush=True)
    df = pd.DataFrame(all_rows)
    S = np.concatenate(all_maps)
    suffix = f"_limit{a.limit}" if a.limit else ""
    for t, tag in TARGETS.items():
        m = (df["type_name"] == t).values
        np.save(OUT_DIR / f"soccermaps_{tag}{suffix}.npy", S[m])
        df[m].reset_index(drop=True).to_parquet(OUT_DIR / f"soccermaps_{tag}{suffix}_idx.parquet")
        print(f"[save] {tag}: {int(m.sum()):,} events", flush=True)
    rep = f"audit/perspective_fix_validation{suffix}.md"
    validation_report(df, rep)
    print(f"[done] {(time.time() - t0) / 60:.1f} min, report {rep}", flush=True)


if __name__ == "__main__":
    main()
