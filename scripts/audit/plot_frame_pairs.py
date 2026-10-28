#!/usr/bin/env python3
"""
plot_frame_pairs.py, freeze frames of paired events side by side
================================================================
Draws the frame of a failed dribble next to the frame of the opponent's tackle at the same moment.
One match from an earlier season and one from 24/25. In each, the first failed dribble that has an opponent Duel frame
at the same time and a visible keeper near a goal. Each row has three panels. Left, the raw frame of the tackle event;
middle, the raw frame of the failed dribble; right, the failed-dribble frame re-oriented following the specification.
Pitches use the mplsoccer statsbomb pitch (origin top left, y axis downwards, 120x80 yards), and by convention the team
that owns the event attacks from left to right.

Inputs   STATSBOMB_DATA_DIR/{season}/{match_id}_events.json and _360.json (read only)
Outputs  outputs/figures/frame_pair_examples.png

Usage
  PYTHONPATH=. python -u scripts/audit/plot_frame_pairs.py

Last modified 2026-10-28
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mplsoccer import Pitch

# Directory with the raw StatsBomb files, one sub-directory per season. Set the
# STATSBOMB_DATA_DIR environment variable; the default is data/statsbomb under the
# repository root. The data are not distributed with this repository (see README).
DATA_DIR = Path(os.environ.get("STATSBOMB_DATA_DIR", "data/statsbomb"))
EXAMPLES = [("11_235_2022_23", 3845186, "22/23 La Liga"), ("11_317_2024_25", 3946398, "24/25 La Liga")]
OUT = Path("outputs/figures/frame_pair_examples.png")
C_MATE, C_OPP, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def pick(season, mid):
    ev = json.load(open(DATA_DIR / season / f"{mid}_events.json", encoding="utf-8"))
    fr = {x["event_uuid"]: x for x in json.load(open(DATA_DIR / season / f"{mid}_360.json", encoding="utf-8"))}
    byid = {e["id"]: e for e in ev}
    for e in ev:
        if e["type"]["name"] != "Dribble" or e["dribble"]["outcome"]["name"] != "Incomplete" or e["id"] not in fr:
            continue
        ff = fr[e["id"]]["freeze_frame"]
        if not any(p["keeper"] and (p["location"][0] < 30 or p["location"][0] > 90) for p in ff):
            continue
        for rid in e.get("related_events", []):
            r = byid.get(rid)
            if r and r["type"]["name"] == "Duel" and rid in fr and r["timestamp"] == e["timestamp"]:
                return e, ff, r, fr[rid]["freeze_frame"]
    raise RuntimeError("no example found")


def draw(ax, pitch, ff, loc, title, note, mirror=False, flip=False):
    pitch.draw(ax=ax)
    for p in ff:
        x, y = p["location"][:2]
        if mirror:
            x, y = 120 - x, 80 - y
        mate = (not p["teammate"]) if flip else p["teammate"]
        pitch.scatter(x, y, ax=ax, s=150 if p["keeper"] else 70, marker="o" if mate else "s",
                      c=C_MATE if mate else C_OPP, edgecolors=INK if p["actor"] else SURFACE,
                      linewidths=2.0 if p["actor"] else 1.0, zorder=3)
        if p["keeper"]:
            pitch.annotate("keeper", (x, y), (min(max(x, 7), 113), y - 5), ax=ax, ha="center", va="bottom", fontsize=8, color=MUTED)
    pitch.scatter(loc[0], loc[1], ax=ax, s=160, marker="x", c=INK, linewidths=2.0, zorder=4)
    pitch.arrows(48, 86, 72, 86, ax=ax, width=1.5, headwidth=5, headlength=5, color=MUTED, clip_on=False)
    ax.text(60, 91, "attacking direction of the event's team (convention)", ha="center", va="top", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=16)
    ax.text(0, -2.5, note, fontsize=8.5, color=MUTED, ha="left", va="bottom")


def main():
    pitch = Pitch(pitch_type="statsbomb", pitch_color=SURFACE, line_color="#b9b8b2", linewidth=1)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.4), facecolor=SURFACE)
    for row, (season, mid, label) in enumerate(EXAMPLES):
        e, ff, r, rf = pick(season, mid)
        old = "2024_25" not in season
        lead = f"{label}, match {mid}, {e['timestamp'][:8]}"
        draw(axes[row, 0], pitch, rf, r["location"],
             f"{lead}\nraw frame of the tackle event ({r['team']['name']}, index {r['index']})",
             "follows the convention, tackler at the event location, own keeper on the left")
        draw(axes[row, 1], pitch, ff, e["location"],
             f"same moment\nraw frame of the failed dribble ({e['team']['name']}, index {e['index']})",
             "coordinates not turned to the dribbler's direction, players on the mirrored side of the event location" if old
             else "coordinates turned, but flags still refer to the tackler's team; the keeper of the attacked goal is flagged teammate")
        draw(axes[row, 2], pitch, ff, e["location"],
             f"re-oriented following the specification\nframe of the failed dribble ({e['team']['name']})",
             "whole frame mirrored (120-x, 80-y), flags unchanged" if old else "coordinates unchanged, all teammate flags inverted",
             mirror=old, flip=not old)
        print(f"[example] {season} {mid} dribble index {e['index']} loc {e['location']} duel index {r['index']} loc {r['location']}", flush=True)
    handles = [Line2D([], [], marker="o", ls="", ms=9, mfc=C_MATE, mec=SURFACE, label="teammate true (flagged teammate in the frame)"),
               Line2D([], [], marker="s", ls="", ms=9, mfc=C_OPP, mec=SURFACE, label="teammate false (flagged opponent in the frame)"),
               Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=INK, mew=2, label="black edge marks the point flagged actor"),
               Line2D([], [], marker="x", ls="", ms=9, mec=INK, mew=2, label="location of the event in the event file")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle("One snapshot given to two events; the frame of the failed dribble is not fully converted as the specification requires", fontsize=14, color=INK, x=0.02, ha="left")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.10, wspace=0.05, hspace=0.42)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
