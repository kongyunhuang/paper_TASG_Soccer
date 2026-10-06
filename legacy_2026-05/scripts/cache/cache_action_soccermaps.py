#!/usr/bin/env python3
"""
Cache SoccerMaps (7×8×12) for 8 action types
============================================================================
Generate SoccerMap spatial tensors from the L1 event cache and the raw StatsBomb 360 JSON files.
v3: adapted to L1_events_v3.parquet, 7 channels A-G (aligned with un-xPass pass success p.4-5).

Channel definitions (see create_soccermap):
  Ch0 (A): teammate positions, Ch1 (B): opponent positions
  Ch2 (C): per-cell distance to ball, Ch3 (D): per-cell distance to goal
  Ch4 (E): sin(angle to ball), Ch5 (F): cos(angle to ball), Ch6 (G): angle to goal

8 action types: Pass, Dribble, Ball Receipt*, Pressure, Shot, Duel, Interception, Ball Recovery

Inputs:
  - data/L1_events_v3.parquet  (event cache)
  - StatsBomb 360 JSON files (Google Drive datos/ directory)

Outputs (one set per action type):
  - data/action_soccermaps_{type}.npy         (N, 7, 8, 12) float32
  - data/action_soccermaps_{type}_idx.parquet  event_id index

Usage:
  PYTHONUNBUFFERED=1 python scripts/cache/cache_action_soccermaps.py

"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ── SoccerMap grid constants ──────────────────────────────────────────────
GRID_WIDTH = 12
GRID_HEIGHT = 8
PITCH_LENGTH = 120.0  # StatsBomb coordinates in yards
PITCH_WIDTH = 80.0
N_CHANNELS = 7  # aligned with un-xPass pass success A-G (verified against original p.4-5)

CACHE_DIR = Path("data")

# Path to the local directory containing the StatsBomb match-level 360 JSON
# files (one folder per season). Set the STATSBOMB_DATA_DIR environment
# variable before running this script.
_data_dir_env = os.environ.get("STATSBOMB_DATA_DIR", "")
if not _data_dir_env:
    raise RuntimeError(
        "STATSBOMB_DATA_DIR is not set. Export it to the local directory "
        "containing the StatsBomb 360 JSON files, e.g.\n"
        "    export STATSBOMB_DATA_DIR=/path/to/statsbomb/datos"
    )
DATA_DIR = Path(_data_dir_env)


def find_360_file(match_id: int) -> Path | None:
    for season_dir in DATA_DIR.iterdir():
        if not season_dir.is_dir() or season_dir.name.startswith("."):
            continue
        candidate = season_dir / f"{match_id}_360.json"
        if candidate.exists():
            return candidate
    return None


def load_360_index(path_360: Path) -> dict[str, dict]:
    with open(path_360, "r", encoding="utf-8") as f:
        frames = json.load(f)
    return {fr["event_uuid"]: fr for fr in frames}


def create_soccermap(
    ball_x: float, ball_y: float,
    teammates_coords: list | None = None,
    opponents_coords: list | None = None,
) -> np.ndarray:
    """Construct a 7-channel SoccerMap (7, 8, 12).

    Aligned with un-xPass pass success channels A-G (verified against original p.4-5):
      Ch0 (A): teammate positions: each teammate's cell += 1
      Ch1 (B): opponent positions: each opponent's cell += 1
      Ch2 (C): per-cell distance to ball: Euclidean distance from cell centre to (ball_x, ball_y)
      Ch3 (D): per-cell distance to goal: Euclidean distance from cell centre to (120, 40)
      Ch4 (E): per-cell sin of angle to ball: sin(arctan2(ball_y - cy, ball_x - cx))
      Ch5 (F): per-cell cos of angle to ball: cos(arctan2(ball_y - cy, ball_x - cx))
      Ch6 (G): per-cell angle to goal: arctan2(40 - cy, 120 - cx)

    Note: the actor is not excluded (un-xPass p.4 wording "attacking and defending team" includes the actor).
    Coordinate system: StatsBomb 120×80 yards.
    """
    smap = np.zeros((N_CHANNELS, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    cell_w = PITCH_LENGTH / GRID_WIDTH    # 10.0
    cell_h = PITCH_WIDTH / GRID_HEIGHT    # 10.0

    def _to_grid(x, y):
        gx = min(max(int(x / cell_w), 0), GRID_WIDTH - 1)
        gy = min(max(int(y / cell_h), 0), GRID_HEIGHT - 1)
        return gx, gy

    # Pre-compute grid cell centre coordinates
    cx = np.arange(GRID_WIDTH) * cell_w + cell_w / 2      # (12,)
    cy = np.arange(GRID_HEIGHT) * cell_h + cell_h / 2     # (8,)
    CX, CY = np.meshgrid(cx, cy)  # both (8, 12)

    # Ch0 (A): teammate positions
    if teammates_coords:
        for p in teammates_coords:
            px, py = float(p[0]), float(p[1])
            if np.isfinite(px) and np.isfinite(py):
                gx, gy = _to_grid(px, py)
                smap[0, gy, gx] += 1.0

    # Ch1 (B): opponent positions
    if opponents_coords:
        for p in opponents_coords:
            px, py = float(p[0]), float(p[1])
            if np.isfinite(px) and np.isfinite(py):
                gx, gy = _to_grid(px, py)
                smap[1, gy, gx] += 1.0

    # Ch2 (C): per-cell distance to ball
    if np.isfinite(ball_x) and np.isfinite(ball_y):
        smap[2] = np.sqrt((CX - ball_x)**2 + (CY - ball_y)**2)

    # Ch3 (D): per-cell distance to goal
    goal_x, goal_y = 120.0, 40.0
    smap[3] = np.sqrt((CX - goal_x)**2 + (CY - goal_y)**2)

    # Ch4 (E): per-cell sin of angle to ball
    if np.isfinite(ball_x) and np.isfinite(ball_y):
        angle_to_ball = np.arctan2(ball_y - CY, ball_x - CX)
        smap[4] = np.sin(angle_to_ball)

    # Ch5 (F): per-cell cos of angle to ball
    if np.isfinite(ball_x) and np.isfinite(ball_y):
        smap[5] = np.cos(angle_to_ball)

    # Ch6 (G): per-cell angle to goal
    smap[6] = np.arctan2(goal_y - CY, goal_x - CX)

    return smap


def process_events_for_match(match_id, event_ids, ball_xs, ball_ys):
    n = len(event_ids)
    smaps = np.zeros((n, N_CHANNELS, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    has_ff = [False] * n

    path_360 = find_360_file(match_id)
    if path_360 is None:
        return smaps, has_ff

    frame_index = load_360_index(path_360)

    for i in range(n):
        frame = frame_index.get(event_ids[i])
        if frame is not None and frame.get("freeze_frame"):
            ff = frame["freeze_frame"]
            teammates = []
            opponents = []
            for p in ff:
                loc = p.get("location")
                if loc is None or len(loc) < 2:
                    continue
                if p.get("teammate", False):
                    teammates.append([float(loc[0]), float(loc[1])])
                else:
                    opponents.append([float(loc[0]), float(loc[1])])
            smaps[i] = create_soccermap(ball_xs[i], ball_ys[i], teammates, opponents)
            has_ff[i] = True

    return smaps, has_ff


def cache_action_type(df_events, action_type):
    df_act = df_events[df_events["type_name"] == action_type].copy()
    print(f"\n{'='*60}")
    print(f"Caching SoccerMaps for: {action_type}")
    print(f"  Total events: {len(df_act):,}")

    df_act = df_act[df_act["has_360"] == True].reset_index(drop=True)
    print(f"  With has_360: {len(df_act):,}")

    match_ids = sorted(df_act["match_id"].unique())
    n_matches = len(match_ids)
    print(f"  Matches: {n_matches}")

    all_smaps = []
    all_event_ids = []
    total_with_ff = 0
    t0 = time.time()

    for i, mid in enumerate(match_ids):
        df_m = df_act[df_act["match_id"] == mid]
        eids = df_m["event_id"].tolist()
        bxs = df_m["location_x"].tolist()
        bys = df_m["location_y"].tolist()

        smaps, has_ff = process_events_for_match(mid, eids, bxs, bys)
        total_with_ff += sum(has_ff)
        all_smaps.append(smaps)
        all_event_ids.extend(eids)

        if (i + 1) % 500 == 0 or i == 0 or i == n_matches - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_matches - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n_matches}] elapsed={elapsed:.0f}s, ETA={eta:.0f}s")

    smaps_all = np.concatenate(all_smaps, axis=0)
    tag = action_type.lower().replace(" ", "_").replace("*", "")
    npy_path = CACHE_DIR / f"action_soccermaps_{tag}.npy"
    idx_path = CACHE_DIR / f"action_soccermaps_{tag}_idx.parquet"

    np.save(npy_path, smaps_all)
    pd.DataFrame({"event_id": all_event_ids}).to_parquet(idx_path, index=False)

    size_mb = npy_path.stat().st_size / (1024 * 1024)
    ff_rate = total_with_ff / len(all_event_ids) * 100
    print(f"  Saved: {npy_path.name} ({size_mb:.0f} MB), shape={smaps_all.shape}")
    print(f"  FF coverage: {total_with_ff:,}/{len(all_event_ids):,} ({ff_rate:.1f}%)")


def main():
    l1_path = CACHE_DIR / "L1_events_v3.parquet"
    if not l1_path.exists():
        raise FileNotFoundError(f"Run prepare_l1_events.py first! Missing {l1_path}")

    print("Loading L1 events (v3) ...")
    df = pd.read_parquet(l1_path)
    print(f"  Total events: {len(df):,}")

    for t in ["Pass", "Dribble", "Ball Receipt*", "Pressure",
              "Shot", "Duel", "Interception", "Ball Recovery"]:
        tag = t.lower().replace(" ", "_").replace("*", "")
        npy_path = CACHE_DIR / f"action_soccermaps_{tag}.npy"
        if npy_path.exists():
            print(f"\n  SKIP {t}: {npy_path.name} already exists")
            continue
        cache_action_type(df, t)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
