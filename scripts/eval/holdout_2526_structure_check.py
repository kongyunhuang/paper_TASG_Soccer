#!/usr/bin/env python3
"""
holdout_2526_structure_check.py, structure check and clean match list of 25/26 (no model involved)
==================================================================================================
Uses only the event table, the cache indices, the raw scan and the provider metadata, and compares 25/26 with 22/23, 23/24
and 24/25 (the 202 clean matches and the other 558 matches separately).
  1  Proxy criteria for the state of each match. The 24/25 state is decided from the teammate flags of three secondary event
     classes in the raw freeze frames; for 25/26 only 29 La Liga matches have scanned frames, and the other 117 cannot be
     judged the same way. Three proxy criteria are listed per match
       1  provider metadata last_updated_360 after 2025-04-14 (every match processed after that date is in the normal
          state, see results/raw_360_check_tables.md)
       2  share of shots (penalties excluded) with frame at least 70% (the same criterion as clean_matches for 24/25)
       3  share of out-of-play passes with frame at least 50% (in the 27 normal-state 24/25 matches whose frames depend on
          the outcome this share is only 6% to 7%, against 88% in the clean matches)
     For the 29 scanned matches the reversed share and the state from the 24/25 rule are also given.
  2  share with frame per task by season; for passes by outcome (complete, incomplete, out, other) and for shots by outcome.
  3  the 5 s Pressure label recomputed on the event table that includes 25/26, positive rate by season; positive rates of
     the other four tasks by season.
  4  mean and standard deviation of the four provider 360 scalars by season (input distributions, not model results).
  5  whether the cache index coverage equals has_360 of the event table (by task and season), and the number of usable
     destination events.
Every grouped table is checked for duplicate keys.

Inputs   data/cache/L1_events_v3.parquet, data/cache/action_soccermaps_*_idx.parquet,
         data/raw_scan/{match_metadata,match_inventory,match_state_2425}.parquet,
         data/raw_scan/events/11_318_2025_26_part.parquet
Outputs  results/holdout_2526_structure_check.md
         data/cache/holdout_2526/match_table_2526.parquet     proxy criteria and frame shares of all 146 matches
         data/cache/holdout_2526/clean_matches_2526.parquet   matches that pass the proxy criteria

Usage
  PYTHONPATH=. python -u scripts/eval/holdout_2526_structure_check.py

Last modified 2026-10-28
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
RAW = Path("data/raw_scan")
OUT_DIR = CACHE / "holdout_2526"
OUT_MD = Path("results/holdout_2526_structure_check.md")
SEASON_ORDER = ["22/23", "23/24", "24/25 clean 202", "24/25 other 558", "25/26"]
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
        raise RuntimeError(f"{what}  duplicate key  {d.unique()[:5]}")


def season_group(df, clean2425):
    s = df["season_dir"].astype(str)
    g = np.where(s.str.contains("2022_23"), "22/23", np.where(s.str.contains("2023_24"), "23/24", np.where(s.str.contains("2025_26"), "25/26", "")))
    g = pd.Series(g, index=df.index)
    m2425 = s.str.contains("2024_25")
    g[m2425 & df["match_id"].astype(int).isin(clean2425)] = "24/25 clean 202"
    g[m2425 & ~df["match_id"].astype(int).isin(clean2425)] = "24/25 other 558"
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

    # Cache index. event_id set per task
    idx = {t: set(pq.read_table(CACHE / f"action_soccermaps_{t}_idx.parquet", columns=["event_id"]).column(0).to_pylist()) for t in TASK_TYPES}
    for t in TASK_TYPES:
        df[f"in_{t}"] = df["event_id"].isin(idx[t])

    L = ["# Hold-out confirmation, structure check of 25/26 (no model involved)", "",
         f"Generated by `scripts/eval/holdout_2526_structure_check.py`, {time.strftime('%Y-%m-%d %H:%M')}. The 202 clean matches of 24/25 come from `summarize_u1_clean.clean_matches()`.", ""]

    # 5  cache coverage against has_360
    rows = []
    for t, tn in TASK_TYPES.items():
        d = df[df["type_name"] == tn]
        for g, h in d.groupby("grp", observed=True):
            rows.append({"Task": t, "Season": g, "Events": len(h), "has_360": int(h["has_360"].sum()), "In cache index": int(h[f"in_{t}"].sum()),
                         "has_360 but not in cache": int((h["has_360"] & ~h[f"in_{t}"]).sum()), "In cache without has_360": int((~h["has_360"].astype(bool) & h[f"in_{t}"]).sum())})
    cov = pd.DataFrame(rows)
    no_dup(zip(cov["Task"], cov["Season"]), "cache coverage table")
    L += ["## 5  Cache index coverage of has_360 in the event table (by task and season)", "", md(cov), ""]
    print("[cov] has_360 but not in cache, total", int(cov["has_360 but not in cache"].sum()), "; in cache without has_360, total", int(cov["In cache without has_360"].sum()), flush=True)

    # 2  share with frame by season and by outcome
    rows = []
    for t, tn in TASK_TYPES.items():
        d = df[df["type_name"] == tn]
        if t == "shot":
            d = d[d["shot_type_name"] != "Penalty"]
        for g, h in d.groupby("grp", observed=True):
            rows.append({"Task": t + (" (excl. penalties)" if t == "shot" else ""), "Season": g, "Events": len(h), "With frame %": pct(h[f"in_{t}"])})
    fr = pd.DataFrame(rows)
    no_dup(zip(fr["Task"], fr["Season"]), "frame share table")
    L += ["## 2  Share with frame per task (from the cache index), by season", "", md(fr.pivot(index="Task", columns="Season", values="With frame %").reset_index()), "",
          md(fr.pivot(index="Task", columns="Season", values="Events").reset_index()), ""]

    p = df[df["type_name"] == "Pass"].copy()
    p["Outcome"] = p["pass_outcome_name"].fillna("complete").replace({"Incomplete": "incomplete", "Out": "out", "Pass Offside": "offside", "Unknown": "unknown", "Injury Clearance": "injury clearance"})
    rows = []
    for (g, o), h in p.groupby(["grp", "Outcome"], observed=True):
        rows.append({"Season": g, "Outcome": o, "Events": len(h), "With frame %": pct(h["in_pass"])})
    po = pd.DataFrame(rows)
    no_dup(zip(po["Season"], po["Outcome"]), "pass outcome frame table")
    L += ["### Share with frame of passes, by outcome", "", md(po.pivot(index="Outcome", columns="Season", values="With frame %").reset_index()), "",
          md(po.pivot(index="Outcome", columns="Season", values="Events").reset_index()), ""]

    s = df[(df["type_name"] == "Shot") & (df["shot_type_name"] != "Penalty")].copy()
    rows = []
    for (g, o), h in s.groupby(["grp", "shot_outcome_name"], observed=True):
        rows.append({"Season": g, "Outcome": o, "Events": len(h), "With frame %": pct(h["in_shot"])})
    so = pd.DataFrame(rows)
    no_dup(zip(so["Season"], so["Outcome"]), "shot outcome frame table")
    L += ["### Share with frame of shots (excl. penalties), by outcome", "", md(so.pivot(index="Outcome", columns="Season", values="With frame %").reset_index()), "",
          md(so.pivot(index="Outcome", columns="Season", values="Events").reset_index()), ""]

    # 3  label positive rates
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
        rows.append({"Season": g, "Pressure 5 s positive rate %": pct(pr["press_y"].fillna(0)), "Pressure events with frame": len(pr),
                     "Pass completion %": pct(pa["pass_outcome_name"].isna()), "Shot goal rate %": pct(sh["shot_outcome_name"] == "Goal"),
                     "Interception success %": pct(ic["interception_outcome_name"].isin(["Won", "Success In Play"])),
                     "Ball Recovery success %": pct(br["ball_recovery_failure"] != True),
                     "Dribble success %": pct(dr["dribble_outcome_name"] == "Complete"),
                     "Tackle success %": pct(tk["duel_outcome_name"].isin(["Won", "Success", "Success In Play", "Success Out"])), "Tackles with frame and label": len(tk),
                     "Usable destination events (frame and end location)": int((pa["pass_end_location_x"].notna() & pa["pass_end_location_y"].notna()).sum())})
    lab = pd.DataFrame(rows)
    no_dup(lab["Season"], "label table")
    L += ["## 3  Positive rate of each task label by season (events with frame only, as in training)", "", md(lab), "",
          "The 5 s Pressure label is recomputed with `train_unified_clean.pressure_labels_5s` on the whole event table including 25/26. Dribble and Tackle use the frame availability of the original, not re-oriented, cache; only the label distribution is relevant here, the tensors themselves cannot be used.", ""]

    # 4  scalar distributions
    rows = []
    for g, h in df[df["has_360"].astype(bool)].groupby("grp", observed=True):
        r = {"Season": g, "Events with frame": len(h)}
        for c in F_SB:
            r[c + " mean"] = round(float(h[c].mean()), 3)
            r[c + " sd"] = round(float(h[c].std()), 3)
            r[c + " missing %"] = pct(h[c].isna())
        rows.append(r)
    sc = pd.DataFrame(rows)
    no_dup(sc["Season"], "scalar table")
    L += ["## 4  Distribution of the provider 360 scalars by season (all events with frame)", "", md(sc), ""]

    # 1  25/26 per match
    meta = pd.read_parquet(RAW / "match_metadata.parquet")
    meta = meta[meta["season_dir"].str.contains("2025_26") & (meta["match_status_360"] == "available")].copy()
    meta["match_id"] = meta["match_id"].astype(int)
    no_dup(meta["match_id"], "25/26 metadata")
    inv = pd.read_parquet(RAW / "match_inventory.parquet")
    inv = inv[inv["season_dir"].str.contains("2025_26")].copy()
    inv["match_id"] = inv["match_id"].astype(int)
    no_dup(inv["match_id"], "25/26 inventory")
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
    no_dup(mt["match_id"], "25/26 per-match table")
    mt = mt.merge(meta[["match_id", "match_date", "last_updated", "last_updated_360"]], on="match_id", how="left")
    mt = mt.merge(inv[["match_id", "events_local", "f360_local", "f360_bytes", "events_bytes"]], on="match_id", how="left")
    assert mt["match_date"].notna().all(), "some matches are missing from the metadata"
    mt["lu360"] = pd.to_datetime(mt["last_updated_360"])
    mt["crit1_lu360_after"] = mt["lu360"] > STATE_DATE
    mt["crit2_shot_frame"] = mt["shot_frame_pct"] >= SHOT_MIN
    mt["crit3_out_pass_frame"] = mt["out_pass_frame_pct"] >= OUT_MIN

    # State of the 29 scanned matches, same rule as 24/25
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
    st["scan_state"] = np.where(st["rev_pct"] >= 80, "reversed", np.where(st["rev_pct"] <= 20, "normal", "mixed"))
    st = st.reset_index()
    no_dup(st["match_id"], "29-match state table")
    mt = mt.merge(st, on="match_id", how="left")
    assert len(scanned) == st["match_id"].nunique(), f"scanned matches {len(scanned)} differ from the state table {st['match_id'].nunique()}"
    mt["scan_state"] = mt["scan_state"].fillna("not scanned")
    mt["clean_proxy"] = mt["crit1_lu360_after"] & mt["crit2_shot_frame"] & mt["crit3_out_pass_frame"] & (mt["scan_state"] != "reversed") & (mt["scan_state"] != "mixed")
    mt = mt.sort_values(["season_dir", "match_date", "match_id"]).reset_index(drop=True)
    mt.to_parquet(OUT_DIR / "match_table_2526.parquet", index=False)
    clean = mt[mt["clean_proxy"]]
    clean[["match_id", "season_dir", "match_date"]].to_parquet(OUT_DIR / "clean_matches_2526.parquet", index=False)

    L += ["## 1  Proxy criteria per 25/26 match", "",
          f"Of the 146 matches, last_updated_360 is after {STATE_DATE:%Y-%m-%d} in {int(mt['crit1_lu360_after'].sum())} (earliest {mt['lu360'].min():%Y-%m-%d}, latest {mt['lu360'].max():%Y-%m-%d}); "
          f"the share of shots with frame is at least {SHOT_MIN:.0f}% in {int(mt['crit2_shot_frame'].sum())} (lowest {mt['shot_frame_pct'].min():.1f}%); "
          f"the share of out-of-play passes with frame is at least {OUT_MIN:.0f}% in {int(mt['crit3_out_pass_frame'].sum())} (lowest {mt['out_pass_frame_pct'].min():.1f}%, median {mt['out_pass_frame_pct'].median():.1f}%).",
          f"The frames of {len(st)} La Liga matches were scanned and judged with the 24/25 rule. State distribution {st['scan_state'].value_counts().to_dict()}, largest reversed share of the three secondary event classes {st['rev_pct'].max():.1f}%, "
          f"mean reversed share per match {st['dribble_incomplete_rev_pct'].mean():.1f}% for Dribble Incomplete and {st['aerial_lost_rev_pct'].mean():.1f}% for Aerial Lost, mean mirrored share of all events with frame {st['mirror_pct_all'].mean():.2f}% per match.",
          f"**{len(clean)} matches** pass the three proxy criteria and the scanned state (La Liga {int((clean['season_dir'].str.startswith('11_')).sum())}, EPL {int((clean['season_dir'].str.startswith('2_')).sum())}); the list is in `data/cache/holdout_2526/clean_matches_2526.parquet`.", "",
          "By match (sorted by league and date)", "",
          md(mt[["match_id", "season_dir", "match_date", "last_updated_360", "n_events", "frame_pct", "shots", "shot_frame_pct", "goal_frame_pct", "nongoal_frame_pct",
                 "out_passes", "out_pass_frame_pct", "incomplete_pass_frame_pct", "complete_pass_frame_pct", "pressures", "interceptions", "ball_recoveries",
                 "scan_state", "rev_pct", "dribble_incomplete_rev_pct", "aerial_lost_rev_pct", "events_local", "f360_local", "clean_proxy"]].round(1)), ""]
    # 24/25 reference. Out-of-play pass frame share of the 27 normal-state matches removed by shot_frame_pct, the basis of the third criterion
    st2425 = pd.read_parquet(RAW / "match_state_2425.parquet")
    st2425["match_id"] = st2425["match_id"].astype(int)
    p2425 = df[df["season_dir"].str.contains("2024_25") & (df["type_name"] == "Pass") & (df["pass_outcome_name"] == "Out")]
    opm = p2425.groupby("match_id")["in_pass"].mean().mul(100).rename("out_pass_frame_pct").reset_index()
    opm = opm.merge(st2425[["match_id", "state", "shot_frame_pct"]], on="match_id", how="left")
    opm["Group"] = np.where(opm["state"] != "normal", "reversed or mixed", np.where(opm["shot_frame_pct"] >= 70, "clean 202", "normal with low shot frame share, 27"))
    ref = opm.groupby("Group")["out_pass_frame_pct"].agg(["count", "min", "median", "max"]).round(1).reset_index()
    L += ["24/25 reference, share of out-of-play passes with frame by match group (the basis of the third criterion)", "", md(ref), ""]
    OUT_MD.write_text("\n".join(L), encoding="utf8")
    print(f"[clean] 25/26 matches passing the proxy criteria {len(clean)} / {len(mt)}", flush=True)
    print(f"[done] {OUT_MD}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
