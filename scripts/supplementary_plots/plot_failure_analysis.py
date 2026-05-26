#!/usr/bin/env python3
"""
plot_failure_analysis: Case Study 6: Failure freeze-frame (visual evidence of covariate shift)
================================================================================================
Identifies the four worst dribble M4 over-confident-wrong events (p>0.85, y=0)
in the 24/25 strict holdout. For each event we collect 22/23 successful dribbles
at the same pitch location (Euclidean < 5 yards), average the SoccerMap (2-channel
teammate/opponent) for each set, and draw a 4-panel x 2-column comparison to
visualise the systematic difference in opponent count and placement (a
visible-to-the-eye signature of the covariate shift).

Inputs:
  data/predictions/dribble_M4_CNN_2ch_relaxed.npz (24/25 only)
  data/L1_events_v3.parquet
  data/action_soccermaps_dribble.npy + _idx.parquet
Outputs:
  data/results_failure_analysis.json
  figures/figS1.png + .pdf

Usage:
  PYTHONPATH=. python scripts/supplementary_plots/plot_failure_analysis.py

"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plotting.plot_utils import (COLOR_BALL, COLOR_OPPONENT, COLOR_TEAMMATE,
                                 draw_pitch, set_paper_style)

CACHE = Path("data")
PRED = CACHE / "predictions"
FIG = Path("figures"); FIG.mkdir(exist_ok=True)

P_OVERCONF_THR = 0.85
N_WORST = 4
LOC_RADIUS_YARDS = 8.0   # widened slightly to 8y to increase the number of 22/23 matches
COMPARE_SEASONS = ["2_235_2022_23", "11_235_2022_23"]  # 22/23 EPL + La Liga


def load_data():
    print("Loading dribble M4 relaxed predictions ...")
    d = np.load(PRED / "dribble_M4_CNN_2ch_relaxed.npz", allow_pickle=True)
    df_p = pd.DataFrame({
        "event_id": d["event_id"].astype(str),
        "y": d["y_test"].astype(int),
        "p": d["probs"].astype(float),
    })
    print(f"  n={len(df_p):,}  pos_rate={df_p['y'].mean():.3f}  mean_p={df_p['p'].mean():.3f}")

    print("Loading L1 events ...")
    df_ev = pd.read_parquet(CACHE / "L1_events_v3.parquet",
                            columns=["event_id", "match_id", "season_dir",
                                     "type_name",
                                     "player_id", "player_name", "team_name",
                                     "location_x", "location_y",
                                     "dribble_outcome_name"])
    df_ev["event_id"] = df_ev["event_id"].astype(str)
    df_ev = df_ev[df_ev["type_name"] == "Dribble"].copy()

    print("Loading SoccerMap cache ...")
    smaps = np.load(CACHE / "action_soccermaps_dribble.npy", mmap_mode="r")
    idx_df = pd.read_parquet(CACHE / "action_soccermaps_dribble_idx.parquet")
    idx_df["event_id"] = idx_df["event_id"].astype(str)
    smap_lookup = {eid: i for i, eid in enumerate(idx_df["event_id"].values)}

    return df_p, df_ev, smaps, smap_lookup


def composite_smap(eids, smap_lookup, smaps):
    """Return mean (2,8,12) over event ids. Returns None if no events."""
    valid_idx = [smap_lookup[e] for e in eids if e in smap_lookup]
    if not valid_idx:
        return None, 0
    arr = np.stack([np.asarray(smaps[i][:2]).astype(np.float32) for i in valid_idx])
    return arr.mean(axis=0), len(valid_idx)


def scatter_density(ax, smap_2ch, x_lim=None, alpha_scale=0.85):
    """Plot per-cell density as colored dots (size ∝ count)."""
    cell_w = 120 / 12
    cell_h = 80 / 8
    for ch, color in [(0, COLOR_TEAMMATE), (1, COLOR_OPPONENT)]:
        for gy in range(8):
            for gx in range(12):
                v = float(smap_2ch[ch, gy, gx])
                if v < 0.05:
                    continue
                cx = (gx + 0.5) * cell_w
                cy = (gy + 0.5) * cell_h
                if x_lim is not None and (cx < x_lim[0] or cx > x_lim[1]):
                    continue
                ax.scatter([cx], [cy], s=80 + 250 * min(v, 2.0),
                           c=color, marker="o",
                           edgecolors="black", linewidths=0.5,
                           alpha=alpha_scale, zorder=3)


def main():
    set_paper_style()
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    df_p, df_ev, smaps, smap_lookup = load_data()

    # join 24/25 worst events
    df_24 = df_p.merge(df_ev, on="event_id", how="left")
    over_conf = df_24[(df_24["p"] > P_OVERCONF_THR) & (df_24["y"] == 0)].copy()
    over_conf = over_conf.dropna(subset=["location_x", "location_y"])
    over_conf = over_conf.sort_values("p", ascending=False).reset_index(drop=True)
    print(f"\nOver-conf-wrong (p>{P_OVERCONF_THR}, y=0): {len(over_conf)} events in 24/25 holdout")
    if len(over_conf) < N_WORST:
        print(f"  WARN: only {len(over_conf)}, will use top-{len(over_conf)}")

    # Prefer attacking-half (x>60) candidates
    over_conf_atk = over_conf[over_conf["location_x"] > 60].reset_index(drop=True)
    print(f"  attacking-half candidates: {len(over_conf_atk)}")

    # 22/23 dribble pool (successes only): used to pre-screen for events
    # with delta opp_local < 0. Loaded temporarily for the pre-screen step.
    df_22_pre = df_ev[df_ev["season_dir"].isin(COMPARE_SEASONS)].copy()
    df_22_pre = df_22_pre[df_22_pre["dribble_outcome_name"] == "Complete"]
    df_22_pre = df_22_pre.dropna(subset=["location_x", "location_y"])

    # Pre-screen: compute each candidate's delta opp_local on the 3x3 grid and
    # keep only those with delta < 0 (i.e. the 24/25 local opponent count is
    # *lower* than the 22/23 same-location successes, consistent with the
    # covariate-shift "fewer defenders" direction).
    print(f"  Pre-screening for delta opp_local < 0 (covariate-shift direction-consistent) ...")
    cell_w_pre, cell_h_pre = 120 / 12, 80 / 8
    selected_idx = []
    for ridx in over_conf_atk.index[:60]:  # check top-60 by P
        row = over_conf_atk.loc[ridx]
        eid = str(row["event_id"])
        if eid not in smap_lookup:
            continue
        x0, y0 = float(row["location_x"]), float(row["location_y"])
        smap_24 = np.asarray(smaps[smap_lookup[eid]][:2]).astype(np.float32)
        gx = int(min(11, max(0, x0 // cell_w_pre)))
        gy = int(min(7, max(0, y0 // cell_h_pre)))
        gx_lo, gx_hi = max(0, gx-1), min(12, gx+2)
        gy_lo, gy_hi = max(0, gy-1), min(8, gy+2)
        opp_24_local = smap_24[1, gy_lo:gy_hi, gx_lo:gx_hi].sum()

        d22 = df_22_pre[
            ((df_22_pre["location_x"] - x0)**2 + (df_22_pre["location_y"] - y0)**2)
            <= LOC_RADIUS_YARDS**2
        ]
        eids22 = [e for e in d22["event_id"].astype(str).tolist() if e in smap_lookup]
        if len(eids22) < 30:  # require at least 30 same-location 22/23 successes for stable statistics
            continue
        smap_22 = np.stack([np.asarray(smaps[smap_lookup[e]][:2]).astype(np.float32)
                             for e in eids22]).mean(axis=0)
        opp_22_local = smap_22[1, gy_lo:gy_hi, gx_lo:gx_hi].sum()
        d_opp = opp_24_local - opp_22_local
        if d_opp < 0:
            selected_idx.append(ridx)
            print(f"    [keep] rank-{len(selected_idx)} {row['player_name']:25s} "
                  f"P={row['p']:.2f} loc=({x0:.0f},{y0:.0f}) "
                  f"delta opp_local={d_opp:+.2f} n22={len(eids22)}")
        if len(selected_idx) >= N_WORST:
            break

    if len(selected_idx) >= N_WORST:
        worst = over_conf_atk.loc[selected_idx[:N_WORST]]
        print(f"  -> Using top-{N_WORST} from attacking-half + delta opp_local<0 pre-screen")
    else:
        print(f"  -> Only {len(selected_idx)} pass pre-screen, falling back to top-{N_WORST} attacking-half")
        worst = over_conf_atk.head(N_WORST)

    # 22/23 Dribble pool (success only)
    df_22 = df_ev[df_ev["season_dir"].isin(COMPARE_SEASONS)].copy()
    df_22 = df_22[df_22["dribble_outcome_name"] == "Complete"]  # success only
    df_22 = df_22.dropna(subset=["location_x", "location_y"])
    print(f"22/23 dribble (success): {len(df_22):,} events")

    panels = []
    for i, row in worst.iterrows():
        x0, y0 = float(row["location_x"]), float(row["location_y"])
        d_22 = df_22[
            ((df_22["location_x"] - x0)**2 + (df_22["location_y"] - y0)**2) <= LOC_RADIUS_YARDS**2
        ]
        eids_22 = d_22["event_id"].astype(str).tolist()
        smap_22, n22 = composite_smap(eids_22, smap_lookup, smaps)

        eid_24 = str(row["event_id"])
        if eid_24 not in smap_lookup:
            print(f"  SKIP {eid_24}: no SoccerMap")
            continue
        smap_24 = np.asarray(smaps[smap_lookup[eid_24]][:2]).astype(np.float32)

        # Compute mean opponent count over the attacking half (for the caption).
        # Opponent channel mean over the right half (gx >= 6).
        opp_24 = smap_24[1, :, 6:].sum()
        opp_22 = smap_22[1, :, 6:].sum() if smap_22 is not None else np.nan
        tm_24 = smap_24[0, :, 6:].sum()
        tm_22 = smap_22[0, :, 6:].sum() if smap_22 is not None else np.nan

        panels.append({
            "rank": i + 1,
            "event_id": eid_24,
            "player_name": str(row.get("player_name", "")),
            "team": str(row.get("team_name", "")),
            "p": float(row["p"]),
            "loc_x": x0, "loc_y": y0,
            "n22_match": int(n22),
            "smap_24": smap_24,
            "smap_22": smap_22,
            "opp_24_atk_half": float(opp_24),
            "opp_22_atk_half": float(opp_22) if smap_22 is not None else None,
            "tm_24_atk_half": float(tm_24),
            "tm_22_atk_half": float(tm_22) if smap_22 is not None else None,
        })

    if not panels:
        print("No panels: abort")
        return

    # -- Save JSON --
    json_panels = []
    for p in panels:
        d = {k: v for k, v in p.items() if k not in ("smap_24", "smap_22")}
        json_panels.append(d)
    out_json = {
        "p_overconf_threshold": P_OVERCONF_THR,
        "loc_radius_yards": LOC_RADIUS_YARDS,
        "compare_seasons": COMPARE_SEASONS,
        "n_overconf_wrong_total": int(len(over_conf)),
        "panels": json_panels,
    }
    out_path = CACHE / "results_failure_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2, default=str)
    print(f"\nSaved {out_path}")

    # -- Plot -- full pitch view; ball location is marked directly on the pitch.
    n_p = len(panels)
    fig, axes = plt.subplots(n_p, 2, figsize=(13, 2.9 * n_p),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.15})
    if n_p == 1:
        axes = np.array([axes])

    for r, panel in enumerate(panels):
        x0, y0 = panel["loc_x"], panel["loc_y"]
        # Compute the opponent/teammate density in a 16x12 yard window around the dribble
        # (8x12 grid -> neighbouring cells).
        cell_w, cell_h = 120 / 12, 80 / 8
        gx = int(min(11, max(0, x0 // cell_w)))
        gy = int(min(7, max(0, y0 // cell_h)))
        # Local 3x3 window around dribble
        gx_lo, gx_hi = max(0, gx-1), min(12, gx+2)
        gy_lo, gy_hi = max(0, gy-1), min(8, gy+2)
        opp_24_local = panel["smap_24"][1, gy_lo:gy_hi, gx_lo:gx_hi].sum()
        tm_24_local = panel["smap_24"][0, gy_lo:gy_hi, gx_lo:gx_hi].sum()
        if panel["smap_22"] is not None:
            opp_22_local = panel["smap_22"][1, gy_lo:gy_hi, gx_lo:gx_hi].sum()
            tm_22_local = panel["smap_22"][0, gy_lo:gy_hi, gx_lo:gx_hi].sum()
        else:
            opp_22_local = tm_22_local = None

        for c, (smap, title_tag, is_24) in enumerate([
            (panel["smap_24"],
             f"24/25 over-confident wrong  (P={panel['p']:.2f}, FAILED)",
             True),
            (panel["smap_22"],
             f"22/23 same-location successes  (n={panel['n22_match']}, mean SoccerMap)",
             False),
        ]):
            ax = axes[r, c]
            draw_pitch(ax)
            # full pitch range
            ax.set_xlim(-3, 123)
            ax.set_ylim(-3, 100)  # extra headroom for title

            if smap is not None:
                scatter_density(ax, smap, x_lim=None)
            else:
                ax.text(60, 40, "No matching 22/23 events",
                        ha="center", va="center", fontsize=12, color="#888")

            # ball
            ax.scatter([x0], [y0],
                       c=COLOR_BALL, marker="D", s=240,
                       edgecolors="black", linewidths=1.4, zorder=5)
            # Circle around the dribble start location (radius LOC_RADIUS_YARDS)
            from matplotlib.patches import Circle
            ax.add_patch(Circle((x0, y0), LOC_RADIUS_YARDS,
                                fill=False, edgecolor="#2C3E50",
                                linewidth=1.8, linestyle="--", alpha=0.9,
                                zorder=4))

            # title above pitch (tightened: was y=95 with ylim_top=100)
            ax.text(60, 92, title_tag, fontsize=10.5, ha="center", va="center",
                    fontweight="bold", color="#1a1a1a")
            if is_24:
                ax.text(60, 85,
                        f"{panel['player_name']} ({panel['team']})  |  "
                        f"local 3×3 cells: opp={opp_24_local:.1f}, tm={tm_24_local:.1f}",
                        fontsize=9, ha="center", va="center", color="#555")
            else:
                if opp_22_local is not None:
                    ax.text(60, 85,
                            f"local 3×3 cells: opp={opp_22_local:.2f}, "
                            f"tm={tm_22_local:.2f}",
                            fontsize=9, ha="center", va="center", color="#555")

            # bottom annotation: local difference (only on 22/23 column)
            # was y=-8 with ylim_bottom=-12
            if c == 1 and opp_22_local is not None:
                d_opp = opp_24_local - opp_22_local
                d_tm = tm_24_local - tm_22_local
                col = "#CC3311" if d_opp > 0 else "#1F4E79"
                ax.text(60, -6,
                        f"24/25 - 22/23   |   delta opp local = {d_opp:+.2f}, "
                        f"delta tm local = {d_tm:+.2f}",
                        fontsize=9.5, ha="center", va="center", color=col,
                        fontweight="bold")
            ax.set_ylim(-9, 94)

    fig.suptitle(
        "Failure analysis: over-confident wrong dribbles (24/25) vs same-location successes (22/23)",
        fontsize=12.5, y=0.915)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOR_TEAMMATE,
               markeredgecolor="black", markersize=10, label="Teammate density"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOR_OPPONENT,
               markeredgecolor="black", markersize=10, label="Opponent density"),
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor=COLOR_BALL,
               markeredgecolor="black", markersize=10, label="Dribble start (ball)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.070))

    fig.tight_layout(rect=[0, 0.035, 1, 0.975])

    png = FIG / "figS1.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "figS1.pdf", bbox_inches="tight")
    print(f"Saved {png}")
    plt.close(fig)

    # Print summary
    print("\nPer-panel summary:")
    for p in panels:
        d_opp = (p["opp_24_atk_half"] - p["opp_22_atk_half"]
                 if p["opp_22_atk_half"] is not None else None)
        print(f"  rank #{p['rank']}: {p['player_name']:>22s}  "
              f"P={p['p']:.2f}  loc=({p['loc_x']:.0f},{p['loc_y']:.0f})  "
              f"n_22_match={p['n22_match']}  "
              f"delta opp_atk_half={d_opp:+.2f}" if d_opp is not None else "  (no 22/23)")


if __name__ == "__main__":
    main()
