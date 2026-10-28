#!/usr/bin/env python3
"""
cache_soccermaps_perspective_fixed.py, re-oriented SoccerMaps and 360 scalars for Dribble and Duel
=================================================================================================
In the StatsBomb 360 data, the freeze frames of failed dribbles and lost aerial duels are given from the opponent's
perspective (see results/frame_perspective_report.md). This script re-orients every frame of these two event types to
the direction of the team that owns the event. The re-orientation rules never use the outcome label.
  Coordinates  if the player flagged as actor is closer to the mirrored location (120-x, 80-y) than to the event location,
               the whole frame is mirrored
  Translation  after mirroring, the whole frame is translated so that the actor lies on the event location (the shift is
               stored in the actor_offset column)
  Flags        teammate flags are not changed. The three rules below are evaluated and stored only as diagnostic columns
               (keeper_swap, twin_swap, prev_swap, flag_swapped).
               In 24/25, 531 matches are in a reversed-flag state. Their failed-event frames cannot be repaired here, and
               those matches are excluded from evaluation by match state.
               keeper  when a keeper near a goal is visible, the own keeper should be on the side with x < 60
               twin    when a neighbouring event has a frame with the same point set, the expected flags follow from that
                       frame's flags and its team (22/23 and 23/24 only)
               prev    otherwise, positional continuity with the frame of the previous own-team event
The four 360 scalars are recomputed from the re-oriented frames instead of using the provider's precomputed sb_ fields.
The reference point is always the actor's position in the frame, not the event location. The actor of the event that owns
a frame is placed at that event's exact location, and the frame of a failed dribble belongs to the opponent's tackle event.
With the dribble location as reference point, the tackler's distance to it would be constant at 0.14 yards, which writes
the outcome into the feature.

Inputs   raw StatsBomb files *_events.json, *_360.json and 00_info_all_matches_*.json under STATSBOMB_DATA_DIR
         (6 full seasons)
Outputs  data/cache/perspective_fixed/soccermaps_{dribble,duel}.npy          (N, 7, 8, 12) float32
         data/cache/perspective_fixed/soccermaps_{dribble,duel}_idx.parquet  event_id, recomputed scalars, decision of every rule
         results/perspective_fix_validation.md                               validation tables of the re-orientation

Usage
  PYTHONPATH=. python -u scripts/cache/cache_soccermaps_perspective_fixed.py
  PYTHONPATH=. python -u scripts/cache/cache_soccermaps_perspective_fixed.py --limit 20   (only 20 matches per season, for a test run)

Last modified 2026-10-28
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
# Frames of these types may themselves be in the wrong perspective, so they are not used as references
UNTRUSTED = {"Dribble", "Ball Receipt*"}
MIRROR = np.array([120.0, 80.0])


def frame_arrays(ff):
    P = np.array([[p["location"][0], p["location"][1]] for p in ff], dtype=float).reshape(-1, 2)
    mate = np.array([bool(p.get("teammate")) for p in ff])
    keeper = np.array([bool(p.get("keeper")) for p in ff])
    actor = np.array([bool(p.get("actor")) for p in ff])
    return P, mate, keeper, actor


def actor_mirrored(P, actor, loc):
    """True if the actor is closer to the mirrored location, None if there is no actor"""
    if not actor.any():
        return None
    a = P[actor][0]
    return bool(((a - (MIRROR - loc)) ** 2).sum() < ((a - loc) ** 2).sum())


def keeper_rule(P, mate, keeper):
    """True if the flags are reversed, False if they are normal, None if no usable keeper is visible"""
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

        # In the 24/25 pipeline the flags of two events that share a snapshot are no longer opposite, so the twin rule is
        # unreliable there (on a test subset it disagreed with the keeper rule on almost all tackle events); it is used only
        # for the seasons of the earlier pipeline
        new_pipeline = "2024_25" in season
        if k_swap is not None:
            swap, rule = k_swap, "keeper"
        elif t_swap is not None and not new_pipeline:
            swap, rule = t_swap, "twin"
        elif p_swap is not None:
            swap, rule = p_swap, "prev"
        else:
            swap, rule = False, "none"
        # Teammate flags are not swapped; the three rule decisions are kept only as diagnostic columns
        # (see results/raw_360_check_tables.md). In 22/23, 23/24 and the normal-state matches of 24/25 the flags are already
        # correct, and swapping would only add errors. In the reversed-state matches of 24/25 (531 matches) the frame of a
        # failed event comes entirely from the opponent's paired event, actor included. These are not handled here; those
        # matches are left out of the clean test set (per-match state in data/raw_scan/match_state_2425.parquet).
        # The whole frame is translated so that the actor lies on the event location. The players of the event that owns a
        # frame are placed at that event's exact location, the others at observed positions. In 22/23 and 23/24 the frame of
        # a failed dribble is owned by the tackle event, so the dribbler sits at an observed position about 1 yard away, while
        # in a successful dribble the dribbler is exactly at the event location. "Actor exactly at the event location" alone
        # reaches AUC 0.943 in those seasons (table T6b in results/raw_360_check_tables.md). The event features contain the
        # exact location and the CNN's distance-to-ball channel reveals the actor position, so the two together give away the
        # outcome. After translation every frame has the actor at the event location and the other players at their relative
        # positions, so all frames share the same structure.
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
    """How well each recomputed scalar alone separates the outcome, in three periods. Matches after 2025-04 have essentially clean raw frames and serve as the reference."""
    from sklearn.metrics import roc_auc_score
    df = df.copy()
    state_path = Path("data/raw_scan/match_state_2425.parquet")
    if state_path.exists():
        st = pd.read_parquet(state_path)
        normal = set(st.loc[st["state"] == "normal", "match_id"].astype(int))
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
    lines = ["# Perspective fix validation", "", "Generated by `scripts/cache/cache_soccermaps_perspective_fixed.py`.", "",
             "mirrored_pct is the share of frames judged mirrored from the actor position and re-oriented. swapped_pct is the share finally judged to have reversed teammate flags (diagnostic only, the flags are not changed).",
             "The four rule_ columns give the share of frames decided by each rule. keeper_says_swapped_pct is the share of frames with a visible keeper that the keeper rule judges reversed.",
             "The last two columns give, on frames with a visible keeper, the agreement of the other two rules with the keeper rule, an estimate of their accuracy when no keeper is visible.", "",
             pd.DataFrame(rows).to_markdown(index=False), "",
             "## Single-feature AUC of the four recomputed scalars", "",
             "The three periods should be close. C contains the matches whose raw frames follow the specification; if A is clearly above C, the re-oriented training data still carry outcome-dependent traces. B contains the matches with reversed flags, which this script does not repair, so higher values are expected there.",
             "The last column, actor_offset_gt_half_yard, indicates whether the actor was more than half a yard from the event location before translation. It separates the outcomes by itself, which is why frames are translated; it is not a model input.", "",
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
    rep = f"results/perspective_fix_validation{suffix}.md"
    validation_report(df, rep)
    print(f"[done] {(time.time() - t0) / 60:.1f} min, report {rep}", flush=True)


if __name__ == "__main__":
    main()
