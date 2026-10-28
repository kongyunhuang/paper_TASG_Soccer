#!/usr/bin/env python3
"""
prepare_l1_events.py, event table from the raw StatsBomb JSON
=============================================================
Parses the StatsBomb event and 360 JSON files directly (no SPADL conversion) into one event table.
  360 scalar features   StatsBomb precomputed fields, not recomputed from the freeze frame
                        (StatsBomb API 360 Frames v2.0.0 specification)
  T8 Pressure label     possession change within the next 2 on-ball events
                        (exPress, Lee et al. 2025, Table 2, p.6)
  un-xPass features     dist_defender_end and nb_opp_in_path for passes and shots
                        (un-xPass, ML-KULeuven/un-xPass, features.py)

Inputs   StatsBomb JSON files *_events.json and *_360.json under STATSBOMB_DATA_DIR, one sub-directory per season
Outputs  data/cache/L1_events_v3.parquet

Usage
  export STATSBOMB_DATA_DIR=/path/to/statsbomb
  PYTHONPATH=. PYTHONUNBUFFERED=1 python scripts/cache/prepare_l1_events.py

Last modified 2026-10-28
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────
# Directory with the raw StatsBomb files, one sub-directory per season. Set the
# STATSBOMB_DATA_DIR environment variable; the default is data/statsbomb under the
# repository root. The data are not distributed with this repository (see README).
DATA_DIR = Path(os.environ.get("STATSBOMB_DATA_DIR", "data/statsbomb"))
OUTPUT_PATH = Path("data/cache/L1_events_v3.parquet")

# On-ball event types for the T8 Pressure label, following exPress (Lee et al. 2025), where a SPADL
# "action" corresponds to an on-ball event with an actual football meaning
ON_BALL_TYPES = {
    "Pass", "Dribble", "Shot", "Ball Receipt*", "Carry",
    "Clearance", "Interception", "Ball Recovery", "Block",
    "Foul Committed", "Foul Won", "Duel", "Dribbled Past",
    "Dispossessed", "Miscontrol", "Goal Keeper", "Pressure",
}


# ── un-xPass style features ───────────────────────────────────────────
# Ported from the un-xPass code (ML-KULeuven/un-xPass, features.py)

def compute_dist_defender_end(
    freeze_frame: list[dict],
    end_x: float, end_y: float,
) -> float:
    """Distance from the pass or shot end point to the nearest opponent.

    Follows the dist[i,1] part of dist_defender() in un-xPass features.py.
      opponents_coo = [(o["x"], o["y"]) for o in ff if not o["teammate"]]
      dist[i,1] = np.amin(sqrt((opp_x - end_x)^2 + (opp_y - end_y)^2))
    """
    opponents = []
    for p in freeze_frame:
        if p.get("teammate", False):
            continue  # the actor also has teammate=True and is excluded here
        loc = p.get("location")
        if loc is None or len(loc) < 2:
            continue
        opponents.append((float(loc[0]), float(loc[1])))

    if not opponents:
        return np.nan

    dists = [math.sqrt((ox - end_x)**2 + (oy - end_y)**2) for ox, oy in opponents]
    return min(dists)


def _get_passing_cone(start, end, dist=1):
    """Three vertices of the passing-path triangle.

    Ported from _get_passing_cone() in un-xPass features.py.
    The cone starts at start and is 2*dist wide at the end point.
    """
    if (start[0] == end[0]) or (start[1] == end[1]):
        slope = 0
    else:
        slope = (end[1] - start[1]) / (end[0] - start[0])

    dy = math.sqrt(dist**2 / (slope**2 + 1))
    dx = -slope * dy

    if start[0] == end[0]:
        dx, dy = dy, dx

    pnt1 = (end[0] + dx, end[1] + dy)
    pnt2 = (end[0] - dx, end[1] - dy)
    return [tuple(start), pnt1, pnt2]


def _is_inside_triangle(pnt, triangle):
    """Whether a point lies inside a triangle.

    Ported from _is_inside_triangle() in un-xPass features.py.
    """
    def _is_right_of(line):
        return (
            (line[1][0] - line[0][0]) * (pnt[1] - line[0][1])
            - (pnt[0] - line[0][0]) * (line[1][1] - line[0][1])
        ) <= 0

    return (
        _is_right_of([triangle[0], triangle[1]])
        and _is_right_of([triangle[1], triangle[2]])
        and _is_right_of([triangle[2], triangle[0]])
    )


def compute_nb_opp_in_path(
    freeze_frame: list[dict],
    start_x: float, start_y: float,
    end_x: float, end_y: float,
    path_width: int = 1,
) -> int:
    """Number of opponents inside the passing-path triangle.

    Follows nb_opp_in_path() in un-xPass features.py, with
      path_width=1 (yards, StatsBomb coordinates)
    """
    if start_x == end_x and start_y == end_y:
        return 0

    opponents = []
    for p in freeze_frame:
        if p.get("teammate", False):
            continue
        loc = p.get("location")
        if loc is None or len(loc) < 2:
            continue
        opponents.append((float(loc[0]), float(loc[1])))

    if not opponents:
        return 0

    triangle = _get_passing_cone([start_x, start_y], [end_x, end_y], path_width)
    return sum(_is_inside_triangle(o, triangle) for o in opponents)


# ── Single-match loading ──────────────────────────────────────────────

def load_match(
    events_path: Path,
    path_360: Path | None,
) -> pd.DataFrame:
    """Parse the events of one events JSON and an optional 360 JSON."""
    with open(events_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    # Index the 360 frames
    frame_index: dict[str, dict] = {}
    match_has_360 = path_360 is not None and path_360.exists()
    if match_has_360:
        with open(path_360, "r", encoding="utf-8") as f:
            frames = json.load(f)
        frame_index = {fr["event_uuid"]: fr for fr in frames}

    rows = []
    for ev in events:
        loc = ev.get("location", [None, None])
        loc_x = float(loc[0]) if isinstance(loc, list) and len(loc) >= 2 and loc[0] is not None else None
        loc_y = float(loc[1]) if isinstance(loc, list) and len(loc) >= 2 and loc[1] is not None else None

        type_name = ev.get("type", {}).get("name")
        pass_data = ev.get("pass", {}) or {}
        shot_data = ev.get("shot", {}) or {}
        dribble_data = ev.get("dribble", {}) or {}
        ball_receipt_data = ev.get("ball_receipt", {}) or {}
        duel_data = ev.get("duel", {}) or {}
        gk_data = ev.get("goalkeeper", {}) or {}
        interception_data = ev.get("interception", {}) or {}

        # Pass end location
        pass_end = pass_data.get("end_location", [None, None]) or [None, None]
        pass_end_x = float(pass_end[0]) if len(pass_end) >= 2 and pass_end[0] is not None else None
        pass_end_y = float(pass_end[1]) if len(pass_end) >= 2 and pass_end[1] is not None else None

        row = {
            # ── Base fields ──
            "event_id": ev.get("id"),
            "match_id": None,
            "index": ev.get("index"),
            "period": ev.get("period"),
            "timestamp": ev.get("timestamp"),
            "minute": ev.get("minute"),
            "second": ev.get("second"),
            "type_id": ev.get("type", {}).get("id"),
            "type_name": type_name,
            "team_id": ev.get("team", {}).get("id"),
            "team_name": ev.get("team", {}).get("name"),
            "player_id": ev.get("player", {}).get("id"),
            "player_name": ev.get("player", {}).get("name"),
            "position_name": ev.get("position", {}).get("name"),
            "location_x": loc_x,
            "location_y": loc_y,
            "duration": ev.get("duration"),
            "under_pressure": ev.get("under_pressure", False) or False,
            "possession": ev.get("possession"),
            "possession_team_id": ev.get("possession_team", {}).get("id"),
            "play_pattern_name": ev.get("play_pattern", {}).get("name"),

            # ── Pass fields ──
            "pass_length": pass_data.get("length"),
            "pass_angle": pass_data.get("angle"),
            "pass_end_location_x": pass_end_x,
            "pass_end_location_y": pass_end_y,
            "pass_outcome_name": pass_data.get("outcome", {}).get("name") if pass_data.get("outcome") else None,
            "pass_height_name": pass_data.get("height", {}).get("name") if pass_data.get("height") else None,
            "pass_body_part_name": pass_data.get("body_part", {}).get("name") if pass_data.get("body_part") else None,
            "pass_cross": bool(pass_data.get("cross", False)),
            "pass_switch": bool(pass_data.get("switch", False)),
            "pass_through_ball": (pass_data.get("technique", {}) or {}).get("name") == "Through Ball",
            "pass_cut_back": bool(pass_data.get("cut_back", False)),
            "pass_success_probability": pass_data.get("pass_success_probability"),
            "pass_recipient_id": (pass_data.get("recipient", {}) or {}).get("id"),

            # ── Ball Receipt ──
            "ball_receipt_outcome_name": ball_receipt_data.get("outcome", {}).get("name") if ball_receipt_data.get("outcome") else None,

            # ── Dribble ──
            "dribble_outcome_name": dribble_data.get("outcome", {}).get("name") if dribble_data.get("outcome") else None,
            "dribble_overrun": bool(dribble_data.get("overrun", False)),
            "dribble_nutmeg": bool(dribble_data.get("nutmeg", False)),

            # ── Shot ──
            "shot_statsbomb_xg": shot_data.get("statsbomb_xg"),
            "shot_outcome_name": shot_data.get("outcome", {}).get("name") if shot_data.get("outcome") else None,
            "shot_type_name": shot_data.get("type", {}).get("name") if shot_data.get("type") else None,
            "shot_body_part_name": shot_data.get("body_part", {}).get("name") if shot_data.get("body_part") else None,

            # ── Duel / Interception / GK / Ball Recovery ──
            "duel_type_name": duel_data.get("type", {}).get("name") if duel_data.get("type") else None,
            "duel_outcome_name": duel_data.get("outcome", {}).get("name") if duel_data.get("outcome") else None,
            "interception_outcome_name": interception_data.get("outcome", {}).get("name") if interception_data.get("outcome") else None,
            "gk_outcome_name": gk_data.get("outcome", {}).get("name") if gk_data.get("outcome") else None,
            "ball_recovery_failure": bool(ev.get("ball_recovery", {}).get("recovery_failure", False)) if ev.get("ball_recovery") else None,

            # ── 360 features, StatsBomb precomputed fields ──
            "has_360": False,
            "sb_distance_to_nearest_defender": None,
            "sb_num_defenders_on_goal_side": None,
            "sb_visible_teammates": None,
            "sb_visible_opponents": None,
            "sb_line_breaking_pass": None,
            "sb_ball_receipt_in_space": None,
            "sb_ball_receipt_exceeds_distance": None,

            # ── un-xPass style features (Pass and Shot only) ──
            "ux_dist_defender_end": None,
            "ux_nb_opp_in_path": None,
        }

        # ── 360 feature extraction ──
        event_id = ev.get("id")
        if match_has_360 and event_id in frame_index:
            frame_data = frame_index[event_id]
            ff = frame_data.get("freeze_frame", [])
            if ff:
                row["has_360"] = True

                # A. StatsBomb precomputed fields, read directly
                row["sb_distance_to_nearest_defender"] = frame_data.get("distance_to_nearest_defender")
                row["sb_num_defenders_on_goal_side"] = frame_data.get("num_defenders_on_goal_side_of_actor")
                row["sb_line_breaking_pass"] = frame_data.get("line_breaking_pass")
                row["sb_ball_receipt_in_space"] = frame_data.get("ball_receipt_in_space")
                row["sb_ball_receipt_exceeds_distance"] = frame_data.get("ball_receipt_exceeds_distance")

                # split visible_player_counts into teammates and opponents
                vpc = frame_data.get("visible_player_counts", [])
                actor_team_id = ev.get("team", {}).get("id")
                for entry in vpc:
                    if entry.get("team_id") == actor_team_id:
                        row["sb_visible_teammates"] = entry.get("count")
                    else:
                        row["sb_visible_opponents"] = entry.get("count")

                # B. un-xPass style features (Pass and Shot only)
                if type_name == "Pass" and pass_end_x is not None and loc_x is not None:
                    row["ux_dist_defender_end"] = compute_dist_defender_end(ff, pass_end_x, pass_end_y)
                    row["ux_nb_opp_in_path"] = compute_nb_opp_in_path(
                        ff, loc_x, loc_y, pass_end_x, pass_end_y, path_width=1
                    )
                elif type_name == "Shot" and loc_x is not None:
                    # for a shot the end point is the goal centre (120, 40)
                    row["ux_dist_defender_end"] = compute_dist_defender_end(ff, 120.0, 40.0)
                    row["ux_nb_opp_in_path"] = compute_nb_opp_in_path(
                        ff, loc_x, loc_y, 120.0, 40.0, path_width=1
                    )

        rows.append(row)

    return pd.DataFrame(rows)


# ── Derived features ──────────────────────────────────────────────────

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived event features."""
    goal_x, goal_y = 120.0, 40.0
    dx = goal_x - df["location_x"]
    dy = goal_y - df["location_y"]
    df["dist_to_goal"] = np.sqrt(dx**2 + dy**2)
    df["angle_to_goal"] = np.arctan2(dy, dx)
    df["angle_to_goal_center"] = np.abs(
        np.arctan2(df["location_y"] - 40.0, 120.0 - df["location_x"])
    )

    # Pass end-point features
    mask_pass = df["pass_end_location_x"].notna()
    dx_end = goal_x - df["pass_end_location_x"]
    dy_end = goal_y - df["pass_end_location_y"]
    df["pass_end_dist_to_goal"] = np.where(mask_pass, np.sqrt(dx_end**2 + dy_end**2), np.nan)
    df["pass_end_angle_to_goal"] = np.where(mask_pass, np.arctan2(dy_end, dx_end), np.nan)
    df["pass_is_progressive"] = np.where(
        mask_pass, (df["dist_to_goal"] - df["pass_end_dist_to_goal"]) >= 10.0, False
    )
    df["pass_lateral_displacement"] = np.where(
        mask_pass, np.abs(df["pass_end_location_y"] - df["location_y"]), np.nan
    )

    return df


# ── T8 Pressure labels ───────────────────────────────────────────────

def compute_pressure_labels(df: pd.DataFrame) -> pd.Series:
    """T8 Pressure to turnover label.

    Follows the "2 actions" definition of exPress (Lee et al. 2025, Table 2, p.6). The label is 1 when
    possession_team_id changes within the next 2 on-ball events.

    On-ball events are ON_BALL_TYPES (meta events excluded).

    The index of df must be 0-based and contiguous (reset_index) before the call.
    """
    df = df.reset_index(drop=True)
    pressure_mask = df["type_name"] == "Pressure"
    pressure_indices = df.index[pressure_mask].values

    all_types = df["type_name"].values
    all_poss = df["possession_team_id"].values
    all_match = df["match_id"].values
    is_onball = np.array([t in ON_BALL_TYPES for t in all_types])

    labels = np.zeros(len(pressure_indices), dtype=int)

    for i, pidx in enumerate(pressure_indices):
        mid = all_match[pidx]
        team = all_poss[pidx]

        # look for the next 2 on-ball events
        onball_count = 0
        for offset in range(1, 20):  # look at most 20 events ahead
            ni = pidx + offset
            if ni >= len(df):
                break
            if all_match[ni] != mid:
                break
            if not is_onball[ni]:
                continue
            onball_count += 1
            if all_poss[ni] != team:
                labels[i] = 1
                break
            if onball_count >= 2:
                break

    return pd.Series(labels, index=pressure_indices)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    season_dirs = sorted(
        [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()]
    )
    print(f"Found {len(season_dirs)} season directories:")
    for d in season_dirs:
        n_matches = len(list(d.glob("*_events.json")))
        print(f"  {d.name}: {n_matches} matches")

    all_dfs = []
    total_matches = 0

    for season_dir in season_dirs:
        event_files = sorted(season_dir.glob("*_events.json"))
        print(f"\n{'='*60}")
        print(f"Processing {season_dir.name} ({len(event_files)} matches)")

        for i, ef in enumerate(event_files):
            match_id = int(ef.stem.replace("_events", ""))
            path_360 = season_dir / f"{match_id}_360.json"

            try:
                df_match = load_match(ef, path_360 if path_360.exists() else None)
                df_match["match_id"] = match_id
                df_match["season_dir"] = season_dir.name
                all_dfs.append(df_match)
                total_matches += 1

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{i+1}/{len(event_files)}] match={match_id}, events={len(df_match)}, elapsed={elapsed:.0f}s")
            except Exception as e:
                print(f"  WARNING failed match {match_id}: {e}")

    print(f"\nConcatenating {total_matches} matches...")
    df = pd.concat(all_dfs, ignore_index=True)

    print("Computing derived features...")
    df = compute_derived_features(df)

    # Save
    df.to_parquet(OUTPUT_PATH, index=False)
    elapsed = time.time() - t0
    size_mb = OUTPUT_PATH.stat().st_size / (1024**2)

    print(f"\n{'='*60}")
    print(f"DONE, StatsBomb precomputed 360 fields + un-xPass features + T8 labels")
    print(f"{'='*60}")
    print(f"  Total matches: {total_matches}")
    print(f"  Total events:  {len(df):,}")
    print(f"  With 360:      {df['has_360'].sum():,} ({df['has_360'].mean():.1%})")
    print(f"  Output:        {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print(f"  Elapsed:       {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Checks ──
    print(f"\n── Key field checks ──")

    # Cross-season consistency of the precomputed 360 features
    print(f"\n  sb_distance_to_nearest_defender by season:")
    for s in sorted(df["season_dir"].unique()):
        sub = df[(df["season_dir"] == s) & (df["has_360"] == True)]
        vals = sub["sb_distance_to_nearest_defender"].dropna()
        print(f"    {s}: n={len(vals):>8,}, mean={vals.mean():.2f}, std={vals.std():.2f}")

    # visible_player_counts check
    print(f"\n  sb_visible_teammates by season:")
    for s in sorted(df["season_dir"].unique()):
        sub = df[(df["season_dir"] == s) & (df["has_360"] == True)]
        vals = sub["sb_visible_teammates"].dropna()
        print(f"    {s}: n={len(vals):>8,}, mean={vals.mean():.2f}")

    # un-xPass feature check (Pass only)
    passes = df[df["type_name"] == "Pass"]
    ux_end = passes["ux_dist_defender_end"].dropna()
    ux_path = passes["ux_nb_opp_in_path"].dropna()
    print(f"\n  un-xPass Pass features:")
    print(f"    dist_defender_end: n={len(ux_end):,}, mean={ux_end.mean():.2f}")
    print(f"    nb_opp_in_path:   n={len(ux_path):,}, mean={ux_path.mean():.2f}")

    # T8 label check
    print(f"\n  T8 Pressure labels (2 on-ball actions):")
    pressure_labels = compute_pressure_labels(df)
    print(f"    Total pressures: {len(pressure_labels):,}")
    print(f"    Positive (turnover): {pressure_labels.sum():,} ({pressure_labels.mean():.3f})")

    # Samples per task
    print(f"\n  Samples per task:")
    print(f"    T1 Pass: {(df['type_name']=='Pass').sum():,}")
    print(f"    T2 Dribble: {(df['type_name']=='Dribble').sum():,}")
    print(f"    T3 Ball Receipt: {(df['type_name']=='Ball Receipt*').sum():,}")
    print(f"    T4 Shot (excl penalty): {((df['type_name']=='Shot') & (df['shot_type_name']!='Penalty')).sum():,}")
    print(f"    T5 Duel: {(df['type_name']=='Duel').sum():,}")
    print(f"    T6 Interception: {(df['type_name']=='Interception').sum():,}")
    print(f"    T7 Ball Recovery: {(df['type_name']=='Ball Recovery').sum():,}")
    print(f"    T8 Pressure: {(df['type_name']=='Pressure').sum():,}")

    print(f"\nColumns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()
