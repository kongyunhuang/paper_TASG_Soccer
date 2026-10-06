#!/usr/bin/env python3
"""
成对事件的定格帧对照图  把失败过人的帧和同一时刻对方铲球的帧并排画出来
============================
旧赛季和 24/25 各取一场，各挑第一个满足条件的失败过人（同一时刻有对方的 Duel 帧，且帧里看得到靠近球门的门将）。
每行三幅，左是铲球事件的原始帧，中是失败过人事件的原始帧，右是按说明书约定摆正后的失败过人帧。
球场一律用 mplsoccer 的 statsbomb 球场（原点左上角，y 轴向下，120x80 码），事件所属球队按约定从左向右进攻。

输入  datos/{赛季目录}/{match_id}_events.json 和 _360.json（只读）
输出  audit/figs/frame_pair_examples.png

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/audit/plot_frame_pairs.py

Last modified 2026-10-06 (public release: data directory from STATSBOMB_DATA_DIR; logic unchanged since 2026-10-02)
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
EXAMPLES = [("11_235_2022_23", 3845186, "22/23 西甲"), ("11_317_2024_25", 3946398, "24/25 西甲")]
OUT = Path("audit/figs/frame_pair_examples.png")
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
            pitch.annotate("门将", (x, y), (min(max(x, 7), 113), y - 5), ax=ax, ha="center", va="bottom", fontsize=8, color=MUTED)
    pitch.scatter(loc[0], loc[1], ax=ax, s=160, marker="x", c=INK, linewidths=2.0, zorder=4)
    pitch.arrows(48, 86, 72, 86, ax=ax, width=1.5, headwidth=5, headlength=5, color=MUTED, clip_on=False)
    ax.text(60, 91, "事件所属球队的进攻方向（约定）", ha="center", va="top", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=16)
    ax.text(0, -2.5, note, fontsize=8.5, color=MUTED, ha="left", va="bottom")


def main():
    pitch = Pitch(pitch_type="statsbomb", pitch_color=SURFACE, line_color="#b9b8b2", linewidth=1)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.4), facecolor=SURFACE)
    for row, (season, mid, label) in enumerate(EXAMPLES):
        e, ff, r, rf = pick(season, mid)
        old = "2024_25" not in season
        lead = f"{label}，比赛 {mid}，{e['timestamp'][:8]}"
        draw(axes[row, 0], pitch, rf, r["location"],
             f"{lead}\n铲球事件的原始帧（{r['team']['name']}，index {r['index']}）",
             "符合约定，铲球人在事件位置，本方门将在左侧")
        draw(axes[row, 1], pitch, ff, e["location"],
             f"同一时刻\n失败过人事件的原始帧（{e['team']['name']}，index {e['index']}）",
             "坐标没有翻到过人方的朝向，球员都在事件位置的镜像一侧" if old
             else "坐标翻了，但队友标记仍以铲球方为准，被攻击球门的门将被标成队友")
        draw(axes[row, 2], pitch, ff, e["location"],
             f"按说明书约定摆正后\n失败过人事件的帧（{e['team']['name']}）",
             "整帧镜像 (120-x, 80-y)，标记不动" if old else "坐标不动，队友标记整帧取反",
             mirror=old, flip=not old)
        print(f"[example] {season} {mid} dribble index {e['index']} loc {e['location']} duel index {r['index']} loc {r['location']}", flush=True)
    handles = [Line2D([], [], marker="o", ls="", ms=9, mfc=C_MATE, mec=SURFACE, label="teammate 为真（帧里标为队友）"),
               Line2D([], [], marker="s", ls="", ms=9, mfc=C_OPP, mec=SURFACE, label="teammate 为假（帧里标为对手）"),
               Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=INK, mew=2, label="黑边是帧里标为 actor 的点"),
               Line2D([], [], marker="x", ls="", ms=9, mec=INK, mew=2, label="事件文件里该事件的坐标")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle("同一个快照给了两个事件，失败过人事件拿到的帧没有按约定完整转换", fontsize=14, color=INK, x=0.02, ha="left")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.10, wspace=0.05, hspace=0.42)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
