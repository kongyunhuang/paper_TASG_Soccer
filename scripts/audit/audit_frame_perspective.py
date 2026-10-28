#!/usr/bin/env python3
"""
audit_frame_perspective.py, perspective audit of the 360 freeze frames
=====================================================================
Checks whether the coordinate orientation and the teammate flags of the 360 freeze frames change with the action outcome.
The StatsBomb 360 specification states that frame coordinates share the orientation of the event they belong to and that
teammate refers to the event's actor. This script checks whether the data follow it, in six parts.
  A  UUID version of the event ids by season, and freeze-frame coverage by event type
  B  freeze-frame coverage of shots in 24/25, and the goal rate of shots with and without a frame
  C  from the cached spatial tensors, players in the 3x3 cells around the event location and around the mirrored location,
     by task, season and outcome
  D  a sample of the raw 360 files. By event type and outcome, whether the actor lies at the mirrored location and whether
     the keeper flags agree with the keeper's half
  E  share of shots with frame by match month in 24/25, and the share of reversed teammate flags in failed dribbles and
     aerial duels
  F  single-feature AUC of the four provider-precomputed 360 scalars on the dribble task, in three periods

Inputs   data/cache/L1_events_v3.parquet
         data/cache/action_soccermaps_{task}.npy and the matching _idx.parquet
         raw StatsBomb files *_events.json and *_360.json under STATSBOMB_DATA_DIR
         data/cache/perspective_fixed/soccermaps_{dribble,duel}_idx.parquet (for the flag table of part E; skipped if missing)
Outputs  results/frame_perspective_report.md

Usage
  PYTHONPATH=. python -u scripts/audit/audit_frame_perspective.py

Last modified 2026-10-28
"""

from __future__ import annotations

import collections
import glob
import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.training.train_all import CACHE_DIR, FULL_SEASONS, TASK_CONFIG, compute_pressure_labels

# Directory with the raw StatsBomb files, one sub-directory per season. Set the
# STATSBOMB_DATA_DIR environment variable; the default is data/statsbomb under the
# repository root. The data are not distributed with this repository (see README).
DATA_DIR = Path(os.environ.get("STATSBOMB_DATA_DIR", "data/statsbomb"))
OUT = Path("results/frame_perspective_report.md")
N_MATCHES_PER_SEASON = 10
MAX_PER_TASK = 80000


def part_a_b(lines):
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet",
                         columns=["event_id", "season_dir", "type_name", "has_360", "shot_type_name", "shot_outcome_name"])
    df = df[df["season_dir"].isin(FULL_SEASONS)]
    df["uuid_ver"] = df["event_id"].str[14]
    t = (df.groupby("season_dir")["uuid_ver"].value_counts(normalize=True).unstack().fillna(0) * 100).round(2)
    lines += ["## A. UUID version of the event ids (share by season, percent)", "", t.to_markdown(), ""]
    types = ["Pass", "Ball Receipt*", "Dribble", "Shot", "Duel", "Interception", "Ball Recovery", "Pressure", "Carry"]
    c = df[df["type_name"].isin(types)].groupby(["type_name", "season_dir"])["has_360"].mean().unstack().mul(100).round(1)
    lines += ["Share of events with a freeze frame, by event type (percent)", "", c.to_markdown(), ""]
    s = df[(df["type_name"] == "Shot") & (df["shot_type_name"] != "Penalty")].copy()
    s["goal"] = (s["shot_outcome_name"] == "Goal").astype(int)
    g = s.groupby("season_dir").apply(lambda x: pd.Series({
        "n": len(x), "has_360_pct": round(100 * x["has_360"].mean(), 1),
        "goal_rate_all": round(x["goal"].mean(), 4),
        "goal_rate_with_frame": round(x.loc[x["has_360"] == True, "goal"].mean(), 4),
        "goal_rate_without_frame": round(x.loc[x["has_360"] != True, "goal"].mean(), 4)}), include_groups=False)
    lines += ["## B. Freeze-frame coverage and goal rate of non-penalty shots", "", g.to_markdown(), ""]


def part_c(lines):
    cols = ["event_id", "match_id", "season_dir", "type_name", "possession_team_id", "location_x", "location_y",
            "pass_outcome_name", "dribble_outcome_name", "ball_receipt_outcome_name", "shot_outcome_name",
            "shot_type_name", "shot_statsbomb_xg", "duel_outcome_name", "interception_outcome_name",
            "ball_recovery_failure"]
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet", columns=cols)
    df = df[df["season_dir"].isin(FULL_SEASONS)].reset_index(drop=True)
    pl = compute_pressure_labels(df)
    rng = np.random.RandomState(0)
    rows = []
    for task, cfg in TASK_CONFIG.items():
        if task == "xg":
            continue
        d = df[df["type_name"] == cfg["type_name"]].copy()
        if "filter_fn" in cfg:
            d = cfg["filter_fn"](d)
        y = pl if task == "pressure" else cfg["label_fn"](d)
        d = d.reset_index(drop=True)
        d["y"] = y
        idx = pd.read_parquet(CACHE_DIR / f"action_soccermaps_{cfg['smap_tag']}_idx.parquet")
        pos = {e: i for i, e in enumerate(idx["event_id"].values)}
        d = d[d["event_id"].isin(pos)]
        if len(d) > MAX_PER_TASK:
            d = d.iloc[np.sort(rng.choice(len(d), MAX_PER_TASK, replace=False))]
        d = d.reset_index(drop=True)
        sm = np.load(CACHE_DIR / f"action_soccermaps_{cfg['smap_tag']}.npy", mmap_mode="r")
        ii = np.array([pos[e] for e in d["event_id"]])
        o = np.argsort(ii)
        S = np.empty((len(d), 2, 8, 12), np.float32)
        S[o] = np.asarray(sm[ii[o]][:, :2])
        gx = np.clip((d["location_x"].values / 10).astype(int), 0, 11)
        gy = np.clip((d["location_y"].values / 10).astype(int), 0, 7)

        def nb(gx, gy):
            out = np.zeros(len(d))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    out += S[np.arange(len(d)), :, np.clip(gy + dy, 0, 7), np.clip(gx + dx, 0, 11)].sum(1)
            return out

        a, b = nb(gx, gy), nb(11 - gx, 7 - gy)
        old = ~d["season_dir"].str.contains("2024_25").values
        for era, m in [("22/23 and 23/24", old), ("24/25", ~old)]:
            for lab in (1, 0):
                k = m & (d["y"].values == lab)
                if k.sum() == 0:
                    continue
                rows.append(dict(task=task, seasons=era, label="positive" if lab else "negative", n=int(k.sum()),
                                 players_near_event=round(a[k].mean(), 2), players_near_mirror=round(b[k].mean(), 2),
                                 share_mirrored_pct=round(100 * (b[k] > a[k] + 1).mean(), 1)))
    lines += ["## C. Players around the event location and around the mirrored location in the spatial tensors",
              "",
              "players_near_event is the number of players in the event's cell and its neighbours (3x3); players_near_mirror is the same count around the mirrored location (120-x, 80-y).",
              "share_mirrored_pct is the share of events with more than one extra player around the mirrored location compared with the event location. At most 80,000 events per task.",
              "", pd.DataFrame(rows).to_markdown(index=False), ""]


ON_BALL = {"Pass", "Dribble", "Shot", "Ball Receipt*", "Carry", "Clearance", "Interception", "Ball Recovery", "Block",
           "Foul Committed", "Foul Won", "Duel", "Dribbled Past", "Dispossessed", "Miscontrol", "Goal Keeper", "Pressure"}


def pressure_turnover(ev, i):
    """Same rule as train_all.compute_pressure_labels. Whether possession changes within the next two on-ball events"""
    team = ev[i]["possession_team"]["id"]
    n = 0
    for j in range(i + 1, min(i + 20, len(ev))):
        if ev[j]["type"]["name"] not in ON_BALL:
            continue
        n += 1
        if ev[j]["possession_team"]["id"] != team:
            return "turnover within 2 actions"
        if n >= 2:
            break
    return "no turnover"


def outcome(e, ev=None, i=None):
    t = e["type"]["name"]
    if t == "Pressure":
        return pressure_turnover(ev, i)
    if t == "Shot":
        return "Goal" if e["shot"]["outcome"]["name"] == "Goal" else "No goal"
    if t == "Ball Recovery":
        return "Failure" if (e.get("ball_recovery", {}) or {}).get("recovery_failure") else "Success"
    if t == "Dribble":
        return e["dribble"]["outcome"]["name"]
    if t == "Duel":
        return (e.get("duel", {}).get("outcome") or {}).get("name", "(no outcome, aerial lost)")
    if t == "Ball Receipt*":
        return (e.get("ball_receipt", {}).get("outcome") or {}).get("name", "Complete")
    if t == "Pass":
        return (e["pass"].get("outcome") or {}).get("name", "Complete")
    if t == "Interception":
        return (e.get("interception", {}).get("outcome") or {}).get("name", "(none)")
    return "-"


def part_d(lines):
    random.seed(7)
    rows = []
    for season in ["2_235_2022_23", "11_235_2022_23", "2_281_2023_24", "11_281_2023_24", "2_317_2024_25", "11_317_2024_25"]:
        st = collections.defaultdict(collections.Counter)
        files = random.sample(sorted(glob.glob(str(DATA_DIR / season / "*_events.json"))), N_MATCHES_PER_SEASON)
        for fe in files:
            ev = json.load(open(fe))
            fr = {x["event_uuid"]: x for x in json.load(open(fe.replace("_events", "_360")))}
            for i, e in enumerate(ev):
                t = e["type"]["name"]
                if t not in ("Dribble", "Duel", "Ball Receipt*", "Pass", "Interception", "Pressure", "Ball Recovery", "Shot") or e["id"] not in fr or "location" not in e:
                    continue
                ff = fr[e["id"]].get("freeze_frame") or []
                act = [p for p in ff if p.get("actor")]
                c = st[(t, outcome(e, ev, i))]
                c["n"] += 1
                if not act:
                    c["no_actor"] += 1
                    continue
                ax, ay = act[0]["location"][0], act[0]["location"][1]
                x, y = e["location"][0], e["location"][1]
                mir = ((ax - (120 - x)) ** 2 + (ay - (80 - y)) ** 2) < ((ax - x) ** 2 + (ay - y) ** 2)
                c["mirrored"] += mir
                for p in ff:
                    if p.get("keeper"):
                        px = 120 - p["location"][0] if mir else p["location"][0]
                        own_side = px < 60
                        c["keeper_consistent" if own_side == bool(p["teammate"]) else "keeper_inconsistent"] += 1
        for (t, o), c in sorted(st.items()):
            if c["n"] < 15:
                continue
            kc, ki = c["keeper_consistent"], c["keeper_inconsistent"]
            rows.append(dict(season=season, event=t, outcome=o, n=c["n"], no_actor=c["no_actor"],
                             actor_at_mirror_pct=round(100 * c["mirrored"] / c["n"], 1),
                             keepers_seen=kc + ki,
                             keeper_flag_inconsistent_pct=round(100 * ki / max(kc + ki, 1), 1)))
    lines += ["## D. Sample of the raw 360 files (10 random matches per season)",
              "",
              "actor_at_mirror_pct is the share of events whose player flagged actor is closer to the mirrored location than to the event location.",
              "keeper_flag_inconsistent_pct is the share of keepers whose half, after re-orienting the coordinates, disagrees with the teammate flag"
              " (the own keeper should be on the side with x < 60 and flagged teammate, the opposing keeper the reverse).",
              "", pd.DataFrame(rows).to_markdown(index=False), ""]


def match_months():
    rows = []
    for s in FULL_SEASONS:
        info = json.load(open(next((DATA_DIR / s).glob("00_info_all_matches_*.json")), encoding="utf-8"))
        rows += [dict(match_id=int(m["match_id"]), month=str(m.get("match_date"))[:7],
                      match_status_360=m.get("match_status_360"),
                      data_version=(m.get("metadata") or {}).get("data_version")) for m in info]
    return pd.DataFrame(rows)


def part_e_f(lines):
    from sklearn.metrics import roc_auc_score
    mm = match_months()
    cols = ["event_id", "match_id", "season_dir", "type_name", "has_360", "shot_type_name", "dribble_outcome_name",
            "sb_distance_to_nearest_defender", "sb_num_defenders_on_goal_side", "sb_visible_teammates", "sb_visible_opponents"]
    df = pd.read_parquet(CACHE_DIR / "L1_events_v3.parquet", columns=cols)
    df = df[df["season_dir"].isin(FULL_SEASONS)].merge(mm, on="match_id")
    meta = mm.merge(df[["match_id", "season_dir"]].drop_duplicates(), on="match_id")
    v = meta.groupby("season_dir").agg(n_matches=("match_id", "size"),
                                       status_360=("match_status_360", lambda x: ",".join(sorted(set(map(str, x))))),
                                       data_version=("data_version", lambda x: ",".join(sorted(set(map(str, x))))))
    lines += ["## E. Changes by match month within 24/25", "", "360 status and data version fields of the match metadata, by season", "", v.to_markdown(), ""]
    s = df[(df["type_name"] == "Shot") & (df["shot_type_name"] != "Penalty") & df["season_dir"].str.contains("2024_25")]
    t = s.groupby("month")["has_360"].agg(shots="size", with_frame_pct=lambda x: round(100 * x.mean(), 1))
    lines += ["Share of non-penalty shots with a freeze frame in 24/25, by match month (both leagues)", "", t.to_markdown(), ""]
    fix = Path("data/cache/perspective_fixed")
    if (fix / "soccermaps_dribble_idx.parquet").exists():
        d = pd.concat([pd.read_parquet(fix / "soccermaps_dribble_idx.parquet"), pd.read_parquet(fix / "soccermaps_duel_idx.parquet")])
        d = d[d["season_dir"].str.contains("2024_25") & d["keeper_swap"].notna()].merge(mm[["match_id", "month"]], on="match_id")
        d["group"] = np.where(d["type_name"] == "Dribble", "Dribble " + d["outcome"],
                              np.where(d["outcome"] == "", "Aerial lost", "Tackle"))
        d["rev"] = d["keeper_swap"].astype(bool)
        t = d.groupby(["month", "group"])["rev"].mean().unstack().mul(100).round(1)
        lines += ["Share of 24/25 frames with a visible keeper whose teammate flags disagree with the keeper's half (percent), by match month", "", t.to_markdown(), ""]
    g = df[(df["type_name"] == "Dribble") & (df["has_360"] == True)].copy()
    g["y"] = (g["dribble_outcome_name"] == "Complete").astype(int)
    g["period"] = np.where(~g["season_dir"].str.contains("2024_25"), "A 22/23 and 23/24",
                           np.where(g["month"] >= "2025-04", "C 24/25 Apr-May", "B 24/25 Aug-Mar"))
    rows = []
    for period, x in g.groupby("period"):
        r = dict(period=period, n=len(x))
        for c in ["sb_distance_to_nearest_defender", "sb_num_defenders_on_goal_side", "sb_visible_teammates", "sb_visible_opponents"]:
            ok = x[c].notna()
            a = roc_auc_score(x["y"][ok], x[c][ok])
            r[c.replace("sb_", "auc_")] = round(max(a, 1 - a), 3)
        rows.append(r)
    lines += ["## F. Single-feature AUC of the provider-precomputed 360 scalars on the dribble task", "",
              "The raw frames of period C are essentially clean and give a reference for the true discriminative power of these scalars.", "", pd.DataFrame(rows).to_markdown(index=False), ""]


def main():
    lines = ["# Freeze-frame perspective audit", "",
             "Generated by `scripts/audit/audit_frame_perspective.py`.", ""]
    part_a_b(lines)
    print("[A,B] done", flush=True)
    part_c(lines)
    print("[C] done", flush=True)
    part_d(lines)
    print("[D] done", flush=True)
    part_e_f(lines)
    print("[E,F] done", flush=True)
    OUT.write_text("\n".join(lines), encoding="utf8")
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
