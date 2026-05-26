#!/usr/bin/env python3
"""
plot_what_if_simulation: Case 7: tactical what-if simulation (mimicking exPress Section 5.1)
==============================================================================================
Selects a high-P real successful dribble (e.g. Traore or another top dribbler),
runs a forward pass through the dribble M4 CNN (2-channel) to obtain the baseline P,
then perturbs the SoccerMap input in two ways:
  - Perturbation 1: add an extra opponent directly in front of the dribble start (close defender)
  - Perturbation 2: add an extra opponent behind the dribble start (recovery runner)
A second forward pass yields P', visualising the marginal effect of the perturbation
on the prediction.

WARNING: F7 caveat: we do not recompute
prepare_l1_events nb_opp_in_path. Only the SoccerMap (CNN input) is altered;
the tabular features are kept as-is, so this is a CNN-only marginal sensitivity
analysis, not a complete counterfactual. Section 7.7 of the paper must make this
limitation explicit.

Inputs:
  data/dribble_cnn_weights.pt
  data/L1_events_v3.parquet
  data/action_soccermaps_dribble.npy + _idx.parquet
  data/results_dribble_ranking.json (used to pick a top dribbler)
Outputs:
  data/results_what_if_simulation.json
  figures/figS7.png + .pdf

Usage:
  PYTHONPATH=. python scripts/supplementary_plots/plot_what_if_simulation.py

"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts.plotting.plot_utils import (COLOR_BALL, COLOR_OPPONENT, COLOR_TEAMMATE,
                                 draw_pitch, set_paper_style)
from scripts.training.train_all import CNNMLPModel
from scripts.diagnostic.gradcam_dribble import prepare_dribble_relaxed

CACHE = Path("data")
FIG = Path("figures"); FIG.mkdir(exist_ok=True)


def perturb_soccermap(smap_2ch, ball_xy, mode):
    """
    Strong perturbation: add multiple opponents around ball_xy and (mode-specific)
    remove a teammate from near ball.

    mode='block_path':  add 2 opp on a 3-cell line in front of ball + remove 1 tm near ball
    mode='surround':    add 3 opp on left/right/front of ball (encircling)
    """
    smap_p = smap_2ch.copy().astype(np.float32)
    bx, by = ball_xy
    cell_w, cell_h = 120 / 12, 80 / 8

    def cell_of(px, py):
        px = float(np.clip(px, 0.0, 120.0))
        py = float(np.clip(py, 0.0, 80.0))
        return int(min(11, max(0, px // cell_w))), int(min(7, max(0, py // cell_h)))

    if mode == "block_path":
        # 5 opp blocking the dribble lane (2x2 grid of cells in front of ball)
        added = []
        for dx in (4.0, 9.0, 14.0):
            for dy in (-3.0, 3.0):
                px, py = bx + dx, by + dy
                gx, gy = cell_of(px, py)
                smap_p[1, gy, gx] += 1.0
                added.append((float(np.clip(px, 0, 120)),
                              float(np.clip(py, 0, 80))))
        # remove all teammate from forward 1-cell area
        gxa, gya = cell_of(bx + 7.0, by)
        before = smap_p[0, gya, gxa]
        if before > 0:
            smap_p[0, gya, gxa] = 0.0
            added.append(("rm_tm",
                          float(np.clip(bx + 7.0, 0, 120)),
                          float(np.clip(by, 0, 80))))
        return smap_p, added
    elif mode == "surround":
        # encircle ball with 6 opponents
        added = []
        for dx, dy in ((6.0, 0.0), (-4.0, 0.0),
                       (3.0, 7.0), (3.0, -7.0),
                       (-3.0, 7.0), (-3.0, -7.0)):
            px, py = bx + dx, by + dy
            gx, gy = cell_of(px, py)
            smap_p[1, gy, gx] += 1.0
            added.append((float(np.clip(px, 0, 120)),
                          float(np.clip(py, 0, 80))))
        return smap_p, added
    else:
        raise ValueError(mode)


def scatter_density(ax, smap_2ch, alpha_scale=0.85):
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
                ax.scatter([cx], [cy], s=80 + 250 * min(v, 2.0),
                           c=color, marker="o",
                           edgecolors="black", linewidths=0.5,
                           alpha=alpha_scale, zorder=3)


def main():
    set_paper_style()
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    print("Preparing dribble relaxed split (re-build X scaler) ...")
    d = prepare_dribble_relaxed()
    Xte = d["Xte"]
    sm_te = d["sm_te"].numpy()  # (n, 2, 8, 12)
    eids_te = np.asarray(d["event_id_te"]).astype(str)
    p_test_lookup = {e: i for i, e in enumerate(eids_te)}

    weights_path = CACHE / "dribble_cnn_weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} missing: run gradcam_dribble.py first")
    print(f"Loading model weights {weights_path}")
    model = CNNMLPModel(Xte.shape[1], cnn_channels=2)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # Pick a borderline-P event (P in [0.45, 0.55]): perturbations are most
    # informative for dribbles near the decision boundary. The baseline must be
    # an actual success in the attacking half by a mid-quality dribbler, ensuring
    # the freeze-frame contains several opponents.
    print("\nSelecting a borderline-P (~0.5) successful dribble for what-if ...")

    df_ev = pd.read_parquet(CACHE / "L1_events_v3.parquet",
                            columns=["event_id", "player_id", "player_name",
                                     "team_name", "location_x", "location_y",
                                     "type_name", "season_dir",
                                     "dribble_outcome_name"])
    df_ev["event_id"] = df_ev["event_id"].astype(str)
    df_p = pd.DataFrame({
        "event_id": eids_te,
        "test_idx": np.arange(len(eids_te)),
    })
    df_join = df_p.merge(
        df_ev[df_ev["type_name"] == "Dribble"], on="event_id", how="left"
    )

    # Forward-pass over the entire test set
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    sm_te_t = torch.tensor(sm_te, dtype=torch.float32)
    with torch.no_grad():
        z_all = model(Xte_t, sm_te_t).numpy()
    p_all = 1.0 / (1.0 + np.exp(-z_all))
    df_join["p"] = p_all[df_join["test_idx"].values]

    # Candidate set: P in [0.45, 0.65], success, attacking half, far from the
    # pitch boundary so the perturbation is not clipped.
    # x in [70, 105], y in [10, 70]: leaves >= 15y of margin for the perturbation.
    df_cand = df_join[
        (df_join["dribble_outcome_name"] == "Complete") &
        (df_join["location_x"] > 70) & (df_join["location_x"] < 105) &
        (df_join["location_y"] > 10) & (df_join["location_y"] < 70) &
        (df_join["p"] >= 0.45) & (df_join["p"] <= 0.65)
    ].copy()

    # Further filter: keep events where the 3x3 cells around the ball contain
    # at least 2 opponents (so the perturbation has room to add or remove markers).
    cell_w, cell_h = 120 / 12, 80 / 8
    valid_idx = []
    for ridx in df_cand.index:
        row = df_cand.loc[ridx]
        ti = int(row["test_idx"])
        x0, y0 = float(row["location_x"]), float(row["location_y"])
        gx = int(min(11, max(0, x0 // cell_w)))
        gy = int(min(7, max(0, y0 // cell_h)))
        gx_lo, gx_hi = max(0, gx-1), min(12, gx+2)
        gy_lo, gy_hi = max(0, gy-1), min(8, gy+2)
        opp_local = sm_te[ti, 1, gy_lo:gy_hi, gx_lo:gx_hi].sum()
        if opp_local >= 2:
            valid_idx.append(ridx)
        if len(valid_idx) >= 50:
            break
    df_t = df_cand.loc[valid_idx].sort_values("p").reset_index(drop=True)
    # Pick the one closest to 0.5
    df_t["dist_from_0.5"] = (df_t["p"] - 0.5).abs()
    df_t = df_t.sort_values("dist_from_0.5").reset_index(drop=True)
    print(f"  Found {len(df_t)} borderline-P (0.45-0.65) successful dribbles "
          f"with >=2 opp in 3x3 around ball")

    if len(df_t) == 0:
        # fallback to high-P Adama
        print("  Fallback: high-P Adama Traore dribble")
        with open(CACHE / "results_dribble_ranking.json") as f:
            rk = json.load(f)
        top1 = rk["M4_CNN_2ch"]["top20"][0]
        pid = int(float(top1["player_id"]))
        df_t = df_join[(df_join["player_id"] == pid) &
                       (df_join["dribble_outcome_name"] == "Complete") &
                       (df_join["location_x"] > 80)].sort_values("p", ascending=False).reset_index(drop=True)

    sel = df_t.iloc[0]
    test_idx = int(sel["test_idx"])
    smap0 = sm_te[test_idx]  # (2, 8, 12)
    x0, y0 = float(sel["location_x"]), float(sel["location_y"])
    p_orig = float(sel["p"])
    print(f"  Selected: {sel['player_name']} ({sel['team_name']})  "
          f"loc=({x0:.1f},{y0:.1f})  P_baseline={p_orig:.3f}")

    # -- Perturb and forward --
    x_tab_t = torch.tensor(Xte[test_idx], dtype=torch.float32).unsqueeze(0)

    panels = []
    panels.append({"name": "Original (baseline)\n(real freeze-frame, success)",
                   "smap": smap0, "added": None, "P": p_orig})

    for mode, label in [
        ("block_path",
         "Perturbation 1: Block-path\n(+6 opp wall, −teammates)"),
        ("surround",
         "Perturbation 2: Surround\n(+6 opp encircling ball)"),
    ]:
        smap_p, added = perturb_soccermap(smap0, (x0, y0), mode=mode)
        smap_p_t = torch.tensor(smap_p, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            z_p = model(x_tab_t, smap_p_t).item()
        p_p = 1.0 / (1.0 + np.exp(-z_p))
        panels.append({"name": label, "smap": smap_p, "added": added,
                       "P": float(p_p)})
        print(f"  {label}: P={p_p:.3f}  (Δ = {p_p - p_orig:+.3f})")

    # -- Save JSON --
    json_panels = []
    for p in panels:
        json_panels.append({"name": p["name"], "P": p["P"],
                            "added": (None if p["added"] is None
                                      else [list(a) for a in p["added"]]),
                            "delta_P_vs_orig": float(p["P"] - p_orig),})
    out_json = {
        "event_id": str(sel["event_id"]),
        "player_name": str(sel["player_name"]),
        "team_name": str(sel["team_name"]),
        "season_dir": str(sel["season_dir"]),
        "ball_xy": [x0, y0],
        "P_baseline": p_orig,
        "panels": json_panels,
        "note": ("F7 caveat: perturbation only modifies SoccerMap "
                 "(CNN input). Tabular ux_nb_opp_in_path is NOT recomputed; this is "
                 "a CNN-only marginal sensitivity, not a complete counterfactual."),
    }
    out_path = CACHE / "results_what_if_simulation.json"
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2, default=str)
    print(f"\nSaved {out_path}")

    # -- Plot --
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4),
                             gridspec_kw={"wspace": 0.06})
    for ax, panel in zip(axes, panels):
        draw_pitch(ax)
        ax.set_xlim(-3, 123)
        ax.set_ylim(-3, 100)
        scatter_density(ax, panel["smap"])
        # ball
        ax.scatter([x0], [y0], c=COLOR_BALL, marker="D", s=240,
                   edgecolors="black", linewidths=1.4, zorder=5)
        # Mark perturbation positions (large X = added opponent / grey circle = removed teammate)
        if panel["added"] is not None:
            for item in panel["added"]:
                if isinstance(item[0], str) and item[0] == "rm_tm":
                    rx, ry = item[1], item[2]
                    ax.scatter([rx], [ry], c="#ffffff", marker="o", s=240,
                               edgecolors=COLOR_TEAMMATE, linewidths=2.5,
                               linestyle="--", zorder=6)
                    ax.text(rx, ry, "x", fontsize=14, ha="center", va="center",
                            color=COLOR_TEAMMATE, fontweight="bold", zorder=7)
                else:
                    px, py = item
                    ax.scatter([px], [py], c=COLOR_OPPONENT, marker="X", s=380,
                               edgecolors="black", linewidths=2.0, zorder=6)
                    ax.annotate("", xy=(px, py), xytext=(x0, y0),
                                arrowprops=dict(arrowstyle="->",
                                                color="#5F5F5F",
                                                linewidth=1.2, alpha=0.5),
                                zorder=4)

        # title (above pitch): use set_title so it clips to axes
        ax.set_title(panel["name"], fontsize=10.5, fontweight="bold",
                     color="#1a1a1a", pad=8)

        # P annotation (below pitch).
        # Semantic: baseline = neutral gray; perturbations = anomaly red
        # (caption emphasizes ΔP>0 is counter-intuitive / spurious correlation,
        # so perturbation outcomes should be visually flagged as anomalies,
        # not as "improvement" via success-implying blue).
        if abs(panel["P"] - p_orig) < 1e-6:
            col = "#222222"   # baseline: neutral dark text
        else:
            col = "#CC3311"   # perturbation: anomaly red (Tol)
        delta = panel["P"] - p_orig
        if abs(delta) < 1e-6:
            label = f"P(success) = {panel['P']:.3f}  (baseline)"
        else:
            label = f"P(success) = {panel['P']:.3f}   (Δ = {delta:+.3f})"
        ax.text(60, -8, label, fontsize=12, ha="center", va="center",
                color=col, fontweight="bold")
        ax.set_ylim(-12, 100)

    fig.suptitle(
        f"What-if simulation:  {sel['player_name']} ({sel['team_name']})  "
        f"signature dribble  (loc x={x0:.0f}, y={y0:.0f})",
        fontsize=12, y=1.0)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOR_TEAMMATE,
               markeredgecolor="black", markersize=10, label="Teammate"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLOR_OPPONENT,
               markeredgecolor="black", markersize=10, label="Opponent (original)"),
        Line2D([0], [0], marker="X", linestyle="", markerfacecolor=COLOR_OPPONENT,
               markeredgecolor="black", markersize=12, label="Added opponent (perturbation)"),
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor=COLOR_BALL,
               markeredgecolor="black", markersize=10, label="Ball / dribble start"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    png = FIG / "figS7.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "figS7.pdf", bbox_inches="tight")
    print(f"Saved {png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
