#!/usr/bin/env python3
"""
analyze_raw_360.py, check tables for the raw 360 data
=====================================================
Computes, from the full scan, season completeness, whether the frame perspective follows the provider's specification,
and leakage indicators before and after re-orientation. Reads only the output of scan_raw_360.py, not the raw files.
Seven parts.
  T1  matches, status, version fields, update times, file times, events per match and share with frame, by season
  T2  share of events and shots with frame by match month in 23/24 and 24/25, the matches with the lowest share,
      whether each 24/25 match is in the reversed-flag or the normal state, and how that relates to the provider's
      last_updated_360 and to the share of shots with frame
  T3  the event cache L1_events_v3.parquet reconciled with the raw files, season by season
  T4  whether the frames of every event class follow the specification (coordinate orientation, teammate flags),
      with three independent rules cross-checked
  T5  origin of the non-compliant frames, their relation to the frames of opponent events at the same time and to
      the team and possession_team fields
  T6  leakage indicators of Dribble, Ball Receipt and Duel before and after re-orientation
  T7  positive rate of the pressure task and frequency of possession_team changes, by season and 24/25 match state
      (event files only, independent of the frames)

Decision rules (none uses the outcome label)
  Mirrored coordinates  the actor in the frame is closer to the mirrored location (120-x, 80-y) than to the event location
  Reversed flags        vpc rule      the number of points with teammate true equals the provider's visible_player_counts
                                      of the other team and differs from that of the own team
                        keeper rule   after re-orienting the coordinates, the half of a keeper near a goal disagrees with
                                      its teammate flag
                        pairing rule  the frame of an opponent event at the same time has the same point set and the same
                                      teammate flag at every point
Conventions  StatsBomb API 360 Frames v2.0.0, p.3

Inputs   data/raw_scan/match_inventory.parquet, match_metadata.parquet, events/*.parquet
         data/cache/L1_events_v3.parquet
Outputs  results/raw_360_check_tables.md
         data/raw_scan/corrected_flags_dribble_duel_br.parquet   per-event decisions for the three task event types
         data/raw_scan/match_state_2425.parquet                  state of every 24/25 match (normal, reversed, mixed)

Usage
  PYTHONPATH=. python -u scripts/audit/analyze_raw_360.py

Last modified 2026-10-28
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
    # vpc rule
    f = np.full(len(d), "na", dtype=object)
    has = ok & (d["vpc_team"] >= 0) & (d["vpc_other"] >= 0)
    tie = has & (d["vpc_team"] == d["vpc_other"])
    f[has & ~tie & (d["n_T"] == d["vpc_team"])] = "ok"
    f[has & ~tie & (d["n_T"] == d["vpc_other"])] = "rev"
    f[has & ~tie & (d["n_T"] != d["vpc_team"]) & (d["n_T"] != d["vpc_other"])] = "other"
    f[tie] = "tie"
    d["vpc_rule"] = pd.Categorical(f)
    # keeper rule. Keepers near a goal, after re-orienting the coordinates
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
    # pairing rule
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
            "Season": s,
            "Matches in metadata": len(m),
            "Event status available": int((m["match_status"] == "available").sum()),
            "360 status available": int((m["match_status_360"] == "available").sum()),
            "collection_status Complete": int((m["collection_status"] == "Complete").sum()),
            "Raw events files": len(i),
            "Raw 360 files": int(i["f360_exists"].sum()),
            "Both files local and scanned": len(full),
            "Events per match, mean (min to max)": f"{full['n_events'].mean():.0f} ({full['n_events'].min():.0f} to {full['n_events'].max():.0f})" if len(full) else "not scanned",
            "360 entries per match, mean (min to max)": f"{full['n_360_entries'].mean():.0f} ({full['n_360_entries'].min()} to {full['n_360_entries'].max()})" if len(full) else "not scanned",
            "Events with frame, mean % (min to max)": f"{full['frame_pct'].mean():.1f}% ({full['frame_pct'].min():.1f} to {full['frame_pct'].max():.1f})" if len(full) else "not scanned",
        })
    L += ["### T1a Matches, status and size per match, by season", "",
          "Source. `00_info_all_matches_*.json` (provider match metadata) and a full scan of every `*_events.json` and `*_360.json`. An event has a frame when the 360 file has an entry for it with a non-empty freeze_frame.",
          "", md(pd.DataFrame(rows)), ""]

    rows = []
    for s in [season_label(x) for x in SEASONS]:
        m, i = meta[(meta["season"] == s) & (meta["match_status"] == "available")], inv[inv["season"] == s]
        lu, lu3 = pd.to_datetime(m["last_updated"]), pd.to_datetime(m["last_updated_360"])
        em = pd.to_datetime(i["events_mtime"], unit="s", utc=True).dt.tz_convert("Europe/Madrid")
        ver = d.loc[d["season_dir"].astype(str).map(season_label) == s, "uuid_ver"].astype(str).value_counts(normalize=True)
        rows.append({
            "Season": s,
            "Match dates": f"{m['match_date'].min()} to {m['match_date'].max()}",
            "data_version": "/".join(sorted(m["data_version"].dropna().unique())),
            "shot_fidelity": "/".join(sorted(m["shot_fidelity_version"].dropna().unique())),
            "xy_fidelity": "/".join(sorted(m["xy_fidelity_version"].dropna().unique())),
            "last_updated range": f"{lu.min():%Y-%m-%d} to {lu.max():%Y-%m-%d}",
            "last_updated_360 range": f"{lu3.min():%Y-%m-%d} to {lu3.max():%Y-%m-%d}",
            "Matches with equal event and 360 update times": f"{100 * (m['last_updated'] == m['last_updated_360']).mean():.0f}%",
            "Local file modification time (Europe/Madrid)": f"{em.min():%Y-%m-%d %H:%M} to {em.max():%Y-%m-%d %H:%M}",
            "Event UUID version": " ".join(f"v{k} {100 * v:.0f}%" for k, v in ver.items() if v > 0),
            "Frame coordinates with one decimal": f"{100 * i['coord_1dec_frac'].mean():.0f}%" if i["coord_1dec_frac"].notna().any() else "not scanned",
        })
    L += ["### T1b Version fields, update times and file times", "",
          "Update times come from the provider metadata (available matches only). The local file modification time is the mtime of the events file and can be taken as an upper bound of the retrieval time.",
          "Coordinate decimals are counted over the first 300 frames of each match.",
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
        rows.append({"Match month": mo, "Matches": g["match_id"].nunique(), "Events": len(g),
                     "All events with frame %": pct(g["has_frame"]),
                     "Shots": int(g["is_shot"].sum()), "Shots with frame %": pct(g.loc[g["is_shot"], "has_frame"]),
                     "One-decimal coordinates %": round(100 * inv2[inv2["match_id"].isin(g["match_id"].unique())]["coord_1dec_frac"].mean(), 0),
                     "UUID v5%": pct(g["uuid_ver"].astype(str) == "5"),
                     "Secondary events mirrored %": pct(u["mir"]),
                     "Secondary events flags reversed % (vpc decidable)": pct(dec["vpc_rule"] == "rev")})
    L += ["### T2a 23/24 and 24/25 by match month (both leagues)", "",
          "Secondary events are six classes, failed dribbles, Aerial Lost, incomplete ball receipts, Foul Won, Dispossessed and Dribbled Past (see T5).",
          "", md(pd.DataFrame(rows)), ""]
    low = inv2[inv2["f360_local"]].sort_values("frame_pct").head(12).merge(meta[["match_id", "match_date", "last_updated_360"]], on="match_id", how="left")
    L += ["### T2b The 12 matches with the lowest share of events with frame (all seasons)", "",
          md(low[["season", "match_id", "match_date", "n_events", "n_360_entries", "frame_pct", "last_updated_360"]].round(1)), ""]


def t2_state_2425(d, meta, scan, L):
    """State of every 24/25 match, decided from three secondary event classes unrelated to the three task labels (Dispossessed, Foul Won, Dribbled Past)"""
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
    pm["state"] = np.where(pm["rev_pct"] >= 80, "reversed", np.where(pm["rev_pct"] <= 20, "normal", "mixed"))
    pm.reset_index().to_parquet(scan / "match_state_2425.parquet", index=False)

    dist = pd.cut(pm["rev_pct"], [-1, 1, 20, 80, 99, 100], labels=["0 to 1", "1 to 20", "20 to 80", "80 to 99", "99 to 100"]).value_counts().sort_index()
    L += ["### T2c State of every 24/25 match", "",
          "Share of the Dispossessed, Foul Won and Dribbled Past events of each match (unrelated to the task labels of the paper) that the vpc rule judges to have reversed flags.",
          "The distribution is bimodal and no match falls between 20% and 80%, so matches above 80% are in the reversed state and matches below 20% in the normal state.", "",
          md(dist.rename("Matches").reset_index().rename(columns={"rev_pct": "Reversed share per match %"})), ""]
    rows = []
    for (lg, st), g in pm.groupby(["league", "state"]):
        rows.append({"League": lg, "State": st, "Matches": len(g),
                     "Match dates": f"{g['match_date'].min()} to {g['match_date'].max()}",
                     "last_updated_360": f"{g['last_updated_360'].min():%Y-%m-%d} to {g['last_updated_360'].max():%Y-%m-%d}",
                     "Shots with frame %, match mean": round(g["shot_frame_pct"].mean(), 1),
                     "Matches with over 85% of shots with frame": int((g["shot_frame_pct"] > 85).sum()),
                     "Dribble Incomplete reversed %, match mean": round(g["dribble_incomplete_rev_pct"].mean(), 1),
                     "Aerial Lost reversed %, match mean": round(g["aerial_lost_rev_pct"].mean(), 1),
                     "Ball Receipt Incomplete reversed %, match mean": round(g["br_incomplete_rev_pct"].mean(), 1)})
    L += [md(pd.DataFrame(rows)), ""]
    bins = pd.to_datetime(["2024-08-01", "2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-04-15", "2025-05-01", "2025-06-01", "2025-10-01"])
    labels = ["2024-08 to 12", "2025-01", "2025-02", "2025-03", "2025-04-01 to 04-14", "2025-04-15 to 04-30", "2025-05", "2025-06 to 09"]
    ct = pd.crosstab(pd.cut(pm["last_updated_360"], bins, labels=labels), pm["state"])
    L += ["State against the provider's last_updated_360 (time of the last update of the match's 360 data)", "",
          md(ct.reset_index().rename(columns={"last_updated_360": "last_updated_360 interval"})), ""]

    v2 = x[x["usable"] & x["vpc_rule"].isin(["ok", "rev"])].copy()
    v2["month"] = v2["match_id"].map(m["match_date"]).str[:7]
    v2["rev"] = (v2["vpc_rule"] == "rev") * 100.0
    sec = ["Dribble Incomplete", "Dribbled Past", "Dispossessed", "Foul Won", "Duel Aerial Lost", "Ball Receipt Incomplete", "50/50 Lost"]
    pv = v2[v2["cls"].isin(sec)].pivot_table(index="cls", columns="month", values="rev", aggfunc="mean", observed=True).round(0).dropna(how="all")
    L += ["### T2d Share of reversed flags of each secondary event class in 24/25, by match month", "", md(pv.reset_index().rename(columns={"cls": "Event class"})), ""]

    s = x[(x["type"].astype(str) == "Shot") & (x["subtype"].astype(str) != "Penalty")].copy()
    s["goal"] = s["outcome"].astype(str) == "Goal"
    s["state"] = s["match_id"].map(pm["state"])
    q = s.groupby(["state", "has_frame"]).agg(n=("goal", "size"), goals=("goal", "sum"), goal_rate=("goal", "mean")).round(4).reset_index()
    q.columns = ["Match state", "Shot has frame", "Non-penalty shots", "Goals", "Goal rate"]
    L += ["### T2e Non-penalty shots in 24/25, frame availability and goal rate, by match state", "", md(q), ""]
    o = s.groupby(["state", s["outcome"].astype(str)]).agg(n=("goal", "size"), frame_pct=("has_frame", lambda z: round(100 * z.mean(), 1))).reset_index()
    o = o.pivot(index="outcome", columns="state", values=["n", "frame_pct"])
    o.columns = [f"{b} state {'shots' if a == 'n' else 'with frame %'}" for a, b in o.columns]
    old_s = d[(d["era"] == "22/23 and 23/24") & (d["type"].astype(str) == "Shot") & (d["subtype"].astype(str) != "Penalty")]
    o["22/23 and 23/24 with frame %"] = old_s.groupby(old_s["outcome"].astype(str))["has_frame"].mean().mul(100).round(1)
    L += ["Share of shots with frame, by shot outcome", "", md(o.reset_index().rename(columns={"outcome": "Shot outcome"})), ""]
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
    c["Matches with identical event and frame counts"] = j.groupby("s", observed=True)["same"].sum()
    c.index = [season_label(s) for s in c.index]
    mt = pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC").tz_convert("Europe/Madrid")
    L += ["### T3 Event cache reconciled with the raw files", "",
          f"The event cache `data/cache/L1_events_v3.parquet` used for training (written {mt:%Y-%m-%d %H:%M}) is built by `scripts/cache/prepare_l1_events.py` directly from the raw files.",
          "The three left columns come from the cache, the three right columns from this scan of the raw files. For 25/26 only locally stored files were scanned, so the right side is lower.",
          "", md(c.reset_index().rename(columns={"index": "Season"})), ""]


# ── T4 T5 ────────────────────────────────────────────────────────────

def t4_compliance(d, L):
    u = d[d["usable"]]
    rows = []
    for (era, cls), g in u.groupby(["era", "cls"], observed=True):
        if len(g) < 200:
            continue
        v = g[g["vpc_rule"].isin(["ok", "rev"])]
        k = g[g["keeper_rule"].isin(["ok", "rev"])]
        rows.append({"Period": era, "Event class": cls, "Events with frame": len(g),
                     "Mirrored %": pct(g["mir"]),
                     "vpc decidable %": pct(g["vpc_rule"].isin(["ok", "rev"])),
                     "Flags reversed % (vpc)": pct(v["vpc_rule"] == "rev"),
                     "Keeper visible %": pct(g["keeper_rule"].isin(["ok", "rev"])),
                     "Flags reversed % (keeper)": pct(k["keeper_rule"] == "rev"),
                     "Actor exactly at event location %": pct(g["d_s"] < 0.01)})
    t = pd.DataFrame(rows)
    bad = t[(t["Mirrored %"] > 5) | (t["Flags reversed % (vpc)"] > 5)]
    L += ["### T4a Event classes that do not follow the specification (mirrored or reversed flags above 5%)", "",
          f"All events with a frame, an event location and an actor in the frame, {len(u):,} in total ({int(d['has_frame'].sum()):,} events with frame). The specification states that coordinates share the orientation of the event and that teammate refers to the event's actor; for compliant classes the three anomaly percentages should be close to 0.",
          "", md(bad.sort_values(["Period", "Events with frame"], ascending=[True, False])), "",
          "### T4b Other event classes (all compliant; the 14 largest classes per period)", "",
          md(t[~t.index.isin(bad.index)].sort_values(["Period", "Events with frame"], ascending=[True, False]).groupby("Period").head(14)), ""]

    # Cross-check of the three rules
    both = u[u["vpc_rule"].isin(["ok", "rev"]) & u["keeper_rule"].isin(["ok", "rev"])]
    r1 = pd.crosstab(both["vpc_rule"].astype(str), both["keeper_rule"].astype(str))
    bp = u[u["vpc_rule"].isin(["ok", "rev"]) & u["pair_rule"].isin(["same_flags", "opposite_flags"])]
    r2 = pd.crosstab([bp["era"], bp["vpc_rule"].astype(str)], bp["pair_rule"].astype(str))
    L += ["### T4c Agreement of the three rules", "",
          "vpc rule against keeper rule (events decidable by both, all classes and seasons)", "", md(r1.reset_index()), "",
          f"Agreement {100 * (both['vpc_rule'].astype(str) == both['keeper_rule'].astype(str)).mean():.2f}%, n = {len(both):,}.", "",
          "vpc rule against pairing rule (events with an opponent paired frame and decidable by vpc). same_flags means that the flags of this frame and of the opponent event's frame agree point by point; by the specification the two teams' frames should have opposite flags.", "",
          md(r2.reset_index()), ""]
    return bad


def t5_mechanism(d, bad, L):
    u = d[d["usable"] & (d["era"] != "25/26")]
    sec = sorted(set(bad["Event class"]))
    rows = []
    for (era, cls), g in u[u["cls"].isin(sec)].groupby(["era", "cls"], observed=True):
        if len(g) < 200:
            continue
        hp = g["pair_rel"].notna()
        top = g.loc[hp, "pair_type"].astype(str).value_counts(normalize=True).head(3)
        rows.append({"Period": era, "Event class": cls, "n": len(g),
                     "team equals possession_team %": pct(g["team_id"] == g["poss_team_id"]),
                     "Has opponent paired frame %": pct(hp),
                     "Paired frame at the same time %": pct(g.loc[hp, "pair_dt"] < 1e-6),
                     "Paired event types (top 3)": ", ".join(f"{k} {100 * v:.0f}%" for k, v in top.items()),
                     "Same orientation as paired frame (not mirrored) %": pct(g.loc[hp, "pair_rel"].astype(str) == "S"),
                     "Same flags as paired frame %": pct(g.loc[hp, "pair_flag_agree"] > 0.9),
                     "Actor is the same point as in paired frame %": pct(g.loc[hp, "pair_actor_same"].astype(bool)),
                     "Mirrored % among events without paired frame": pct(g.loc[~hp, "mir"]),
                     })
    L += ["### T5a Where the frames of non-compliant events come from", "",
          "A paired frame is a frame of an opponent event at the same time (or within 4 indices) whose point set coincides with this frame (global mirroring and translation allowed).",
          "By the specification, when one snapshot yields a frame for each team, the two frames should be mirror images of each other with opposite flags at every point.",
          "", md(pd.DataFrame(rows).sort_values(["Period", "n"], ascending=[True, False])), ""]

    # Control. Compliant event classes of the other team
    ctrl = ["Pressure", "Duel Tackle", "Foul Committed", "Interception", "Dribble Complete", "Block", "Ball Recovery"]
    rows = []
    for (era, cls), g in u[u["cls"].isin(ctrl)].groupby(["era", "cls"], observed=True):
        hp = g["pair_rel"].notna()
        rows.append({"Period": era, "Event class": cls, "n": len(g),
                     "team equals possession_team %": pct(g["team_id"] == g["poss_team_id"]),
                     "Mirrored %": pct(g["mir"]),
                     "Flags reversed % (vpc)": pct(g.loc[g["vpc_rule"].isin(["ok", "rev"]), "vpc_rule"] == "rev"),
                     "Has opponent paired frame %": pct(hp),
                     "Mirror image of paired frame %": pct(g.loc[hp, "pair_rel"].astype(str) == "M"),
                     "Same flags as paired frame %": pct(g.loc[hp, "pair_flag_agree"] > 0.9)})
    L += ["### T5b Control, compliant event classes and possession_team", "",
          "If the frame perspective followed possession_team, events whose team differs from possession_team (pressures, tackles, fouls, interceptions) would be mirrored as a whole. They are not.",
          "", md(pd.DataFrame(rows).sort_values(["Period", "n"], ascending=[True, False])), ""]

    # Within the non-compliant classes, split by whether team equals possession_team
    rows = []
    for (era, cls), g in u[u["cls"].isin(sec)].groupby(["era", "cls"], observed=True):
        if len(g) < 200:
            continue
        for same, h in g.groupby(g["team_id"] == g["poss_team_id"]):
            v = h[h["vpc_rule"].isin(["ok", "rev"])]
            rows.append({"Period": era, "Event class": cls, "team equals possession_team": "yes" if same else "no", "n": len(h),
                         "Mirrored %": pct(h["mir"]), "Flags reversed % (vpc)": pct(v["vpc_rule"] == "rev")})
    L += ["### T5c Non-compliant classes split by whether team equals possession_team", "",
          md(pd.DataFrame(rows).sort_values(["Period", "Event class"])), ""]

    # Direction of the pairing. Which side's frame is the original
    p = u[u["pair_rel"].notna() & (u["pair_dt"] < 1e-6) & (u["pair_flag_agree"] > 0.9) & u["vpc_rule"].isin(["ok", "rev"])]
    q = p.groupby(["era", "cls", "pair_type"], observed=True).agg(n=("idx", "size"), rev=("vpc_rule", lambda s: 100 * (s == "rev").mean())).reset_index()
    q = q[q["n"] >= 300].sort_values(["era", "n"], ascending=[True, False]).round(1)
    q.columns = ["Period", "Event class", "Paired event type", "n", "Event flags reversed % (vpc)"]
    L += ["### T5d In pairs with identical flags, which side is reversed", "",
          "When the two teams' frames have identical flags, one of them must be reversed. The table gives, by event class and paired event type, the share of events that the vpc rule judges reversed; close to 100 is the secondary side, close to 0 the primary side.",
          "", md(q), ""]


    # Who the actor is in frames with reversed flags
    rows = []
    for (era, cls), g in u[u["cls"].isin(sec) & (u["vpc_rule"] == "rev")].groupby(["era", "cls"], observed=True):
        if len(g) < 200:
            continue
        rows.append({"Period": era, "Event class": cls, "Frames judged reversed by vpc": len(g),
                     "Actor has teammate true %": pct(g["act_T"].astype(bool)),
                     "Actor is the same point as the actor of the opponent paired frame %": pct(g["pair_actor_same"].astype(float).fillna(0) > 0.5),
                     "Teammate count equals provider own-team count after inverting all flags (actor included) %": pct(g["n_pts"] - g["n_T"] == g["vpc_team"]),
                     "Teammate count equals provider own-team count after inverting with the actor kept as teammate %": pct(g["n_pts"] - g["n_T"] + 1 == g["vpc_team"])})
    L += ["### T5e In frames with reversed flags, who is the point flagged actor", "",
          "In reversed frames the point flagged actor is the same point as the actor of the opponent paired frame, that is, the opponent player rather than the player of this event.",
          "The actor flag must therefore be inverted together with all other flags when re-orienting. The last two columns check this against the provider's own visible_player_counts.",
          "", md(pd.DataFrame(rows).sort_values(["Period", "Frames judged reversed by vpc"], ascending=[True, False])), ""]


# ── T6 ───────────────────────────────────────────────────────────────

SECONDARY_WHEN_SAME_FLAGS = {"Dribbled Past", "Dispossessed", "Foul Won", "Duel Aerial Lost", "Ball Receipt Incomplete", "50/50 Lost"}


def decide_flip(g):
    """Final decision on reversed flags. vpc first, then keeper, then pairing with a class prior. Returns flip and the rule used"""
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
    u["period"] = np.where(u["era"] == "24/25", "24/25 " + u["state2425"].astype(str) + "-state matches", u["era"])
    tasks = {
        "Dribble": (u["type"].astype(str) == "Dribble", lambda g: (g["outcome"].astype(str) == "Complete").astype(int).values),
        "Ball Receipt": (u["type"].astype(str) == "Ball Receipt*", lambda g: (g["outcome"].astype(str) == "").astype(int).values),
        "Duel (incl. Aerial Lost)": (u["type"].astype(str) == "Duel", lambda g: g["outcome"].astype(str).isin({"Won", "Success In Play"}).astype(int).values),
        "Duel, Tackle only": ((u["type"].astype(str) == "Duel") & (u["subtype"].astype(str) == "Tackle"), lambda g: g["outcome"].astype(str).isin({"Won", "Success In Play"}).astype(int).values),
        "Pass (control)": (u["type"].astype(str) == "Pass", lambda g: (g["outcome"].astype(str) == "").astype(int).values),
        "Interception (control)": (u["type"].astype(str) == "Interception", lambda g: g["outcome"].astype(str).isin({"Won", "Success In Play"}).astype(int).values),
    }
    rows_a, rows_b, rows_c, keep = [], [], [], []
    for task, (mask, lab) in tasks.items():
        g = u[mask].copy()
        g["y"] = lab(g)
        flip, rule = decide_flip(g)
        g["flip"], g["flip_rule"] = flip, rule
        mir = g["mir"].values
        # Cell counts after re-orientation. Mirroring selects _s or _m, reversed flags select T or O
        def pick(prefix):
            Ts, Os, Tm, Om = (g[f"{prefix}{c}"].values.astype(float) for c in ("T_s", "O_s", "T_m", "O_m"))
            T = np.where(mir, Tm, Ts)
            O = np.where(mir, Om, Os)
            return np.where(flip, O, T), np.where(flip, T, O)
        g["cT_fix"], g["cO_fix"] = pick("c")
        g["nT_fix"], g["nO_fix"] = pick("n")
        # Distance from the actor point in the frame to the nearest player of the other team. When two players are each other's nearest,
        # this distance does not depend on who is the actor, so for frames with reversed flags (the actor is the opponent) inverting the
        # whole frame still gives the original-flag distance from the actor to the nearest point with teammate false.
        # The second quantity mimics inverting while keeping the actor as a teammate; it then measures the distance from the opponent
        # player to his own teammate.
        g["dopp_fix"] = g["dO_act"]
        g["dopp_keep_actor"] = np.where(flip, g["dT_act"], g["dO_act"])
        if task in ("Dribble", "Ball Receipt", "Duel (incl. Aerial Lost)"):
            keep.append(g[["season_dir", "match_id", "idx", "type", "subtype", "outcome", "period", "y", "mir", "flip", "flip_rule",
                           "vpc_rule", "keeper_rule", "pair_rule", "cT_s", "cO_s", "cT_fix", "cO_fix"]])
        groups = list(g.groupby("period")) + [("24/25 all matches", g[g["era"] == "24/25"])]
        for era, h in groups:
            y = h["y"].values
            rows_a.append({"Task": task, "Period": era, "n": len(h), "Positive rate": round(y.mean(), 3),
                           "Negatives mirrored %": pct(h.loc[h["y"] == 0, "mir"]), "Positives mirrored %": pct(h.loc[h["y"] == 1, "mir"]),
                           "Negatives flags reversed %": pct(h.loc[h["y"] == 0, "flip"]), "Positives flags reversed %": pct(h.loc[h["y"] == 1, "flip"]),
                           "Flags undecidable %": pct(h["flip_rule"] == "none")})
            rows_b.append({"Task": task, "Period": era,
                           "Teammates in ball cell, raw": auc(y, h["cT_s"].values.astype(float)),
                           "Teammates in ball cell, re-oriented": auc(y, h["cT_fix"].values),
                           "Opponents in ball cell, raw": auc(y, h["cO_s"].values.astype(float)),
                           "Opponents in ball cell, re-oriented": auc(y, h["cO_fix"].values),
                           "Teammates in 3x3, raw": auc(y, h["nT_s"].values.astype(float)),
                           "Teammates in 3x3, re-oriented": auc(y, h["nT_fix"].values),
                           "Opponents in 3x3, raw": auc(y, h["nO_s"].values.astype(float)),
                           "Opponents in 3x3, re-oriented": auc(y, h["nO_fix"].values),
                           "Provider scalar, distance to nearest defender": auc(y, h["sb_dnd"].values.astype(float)),
                           "Recomputed actor to nearest opponent distance, whole frame inverted": auc(y, h["dopp_fix"].values.astype(float)),
                           "Same, actor kept as teammate when inverting": auc(y, h["dopp_keep_actor"].values.astype(float)),
                           "Provider scalar, defenders on goal side": auc(y, h["sb_ngs"].values.astype(float)),
                           "Actor exactly at event location": auc(y, (h["d_s"].values < 0.01).astype(float))})
            for yy, name in ((1, "positive"), (0, "negative")):
                k = h[h["y"] == yy]
                rows_c.append({"Task": task, "Period": era, "Label": name, "n": len(k),
                               "Players in 3x3 around event, raw": round((k["nT_s"] + k["nO_s"]).mean(), 2),
                               "Players in 3x3 around mirrored location, raw": round((k["nT_m"] + k["nO_m"]).mean(), 2),
                               "Players in 3x3 around event, re-oriented": round((k["nT_fix"] + k["nO_fix"]).mean(), 2),
                               "Teammates in ball cell, raw": round(k["cT_s"].mean(), 2), "Teammates in ball cell, re-oriented": round(k["cT_fix"].mean(), 2),
                               "Opponents in ball cell, raw": round(k["cO_s"].mean(), 2), "Opponents in ball cell, re-oriented": round(k["cO_fix"].mean(), 2)})
    L += ["### T6a Mirrored and reversed-flag shares by task and label (all events)", "",
          "The final reversed-flag decision uses the vpc rule, then the keeper rule, then the pairing rule; events where none of the three applies are undecidable and treated as not reversed.",
          "24/25 is split into two parts by the match state of T2c.",
          "", md(pd.DataFrame(rows_a)), "",
          "### T6b Single-feature AUC before and after re-orientation", "",
          "AUC is max(a, 1-a). The ball cell is the 10x10-yard cell of the event location, the same as the 8x12 training grid. Re-orientation mirrors mirrored frames back and inverts all flags of frames with reversed flags.",
          "Provider scalars are precomputed fields of the 360 files and cannot be re-oriented, so only their raw AUC is listed. The two recomputed columns use the actor point in the frame as reference and are computed from the re-oriented frames; the second mimics keeping the actor as a teammate when inverting (T5e shows why this is wrong).",
          "The last column is a residual cue that re-orientation cannot remove. The frames of the 24/25 normal-state matches follow the specification and give the reference level of each AUC without perspective problems.",
          "", md(pd.DataFrame(rows_b)), "",
          "### T6c Mean player counts around the event, before and after re-orientation", "", md(pd.DataFrame(rows_c)), ""]
    pd.concat(keep, ignore_index=True).to_parquet(scan / "corrected_flags_dribble_duel_br.parquet", index=False)


# ── T7 ───────────────────────────────────────────────────────────────

def t7_pressure(d, L):
    """Pressure labels with the definition of prepare_l1_events (possession_team changes within the next 2 on-ball events); the function is imported from there"""
    from scripts.cache.prepare_l1_events import compute_pressure_labels
    x = d[d["era"] != "25/26"][["season", "match_id", "idx", "type", "team_id", "poss_team_id", "state2425"]].copy()
    x = x.sort_values(["match_id", "idx"]).reset_index(drop=True)
    x["type_name"] = x["type"].astype(str)
    x["possession_team_id"] = x["poss_team_id"]
    x["grp"] = x["season"].astype(str) + np.where(x["state2425"] != "", " " + x["state2425"].astype(str) + "-state matches", "")
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
        rows.append({"Group": g, "Matches": nm, "Pressures": len(h), "Positive rate": round(h["y"].mean(), 4),
                     "Share with pressing team equal to possession_team": round(h["own"].mean(), 4),
                     "Events per match": round(len(e) / nm), "possession_team changes per match": round(e["chg"].sum() / nm, 1)})
    L += ["### T7 Positive rate of the pressure task and frequency of possession changes", "",
          "The label function is `compute_pressure_labels` in `scripts/cache/prepare_l1_events.py`. It uses only the event types, their order and possession_team from the event files, not the freeze frames.",
          "", md(pd.DataFrame(rows)), ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="data/raw_scan")
    ap.add_argument("--out", default="results/raw_360_check_tables.md")
    a = ap.parse_args()
    scan = Path(a.scan)
    inv = pd.read_parquet(scan / "match_inventory.parquet")
    meta = pd.read_parquet(scan / "match_metadata.parquet")
    d = add_derived(load_events(scan))
    print(f"[derived] {len(d):,} events", flush=True)
    L = ["# Raw 360 data check tables", "",
         f"Generated by `scripts/audit/analyze_raw_360.py` from the full scan in `{scan}`, {pd.Timestamp.now():%Y-%m-%d %H:%M}.",
         "Scan script `scripts/audit/scan_raw_360.py`; raw data read from STATSBOMB_DATA_DIR.", ""]
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
