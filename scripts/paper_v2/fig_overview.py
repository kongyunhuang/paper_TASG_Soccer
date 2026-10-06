#!/usr/bin/env python3
"""
图 fig_overview  方法示意图，把作者 5 月手画的 Figure 1（the hand-drawn Figure 1 of the manuscript）复刻成矢量版
============================
布局、配色、字体风格和框里的文字照原图，按 2026-10-04 负责人转达的作者要求改四处  Loss head 换成七个结果任务的 BCE 和落点 96 格 softmax
（从 32 单元层另出一个 Linear 32 to 96），Concatenate 框加任务嵌入 e_t，记号写 h_e 和 h_s，字号缩到版心后不小于 7 pt。
结构和层宽从 scripts/training/train_unified_clean.py 的 UnifiedGatingClean 实例读出（arch()，带断言）。
图里的真实数据都来自同一次传球事件（EVENT_ID，24/25 的 202 场干净比赛之一）
  球场上的球员点  Google Drive datos 里该场的 360 原始 json，mplsoccer 画，StatsBomb 坐标 y 轴向下
  底部七个通道   data/cache/action_soccermaps_pass.npy 里该事件的 7×8×12 张量，每个通道各自归一到 0 到 1
  门控柱状图     u1 种子 3 的权重（data/cache/u1_clean/u1_s3.pt，带训练时的标量标准化参数）对该事件落点任务前向一次得到的 64 维门控；
                 它的均值和 u1_s3.npz 里存的该事件门控均值核对，一致才画
原来的方框版示意图脚本移到 scripts/paper_v2/_archive/fig_overview_boxes_2026-10-04.py。

输入  data/cache/L1_events_v3.parquet，data/cache/action_soccermaps_pass{.npy,_idx.parquet}，data/cache/u1_clean/u1_s3.{pt,npz}，
      Google Drive datos/{season}/{match_id}_360.json
输出  02_manuscript/figures/v2/fig_overview.{pdf,png}，scripts/paper_v2/out/fig_overview_sources.json 和 fig_overview_event.json

Usage
  conda activate kronos
  cd 01_dev_workspace && PYTHONPATH=. python -u scripts/paper_v2/fig_overview.py

Last modified 2026-10-04
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from scripts.paper_v2.common import CM, DEV, OUT_DIR, Registry, save_fig
from scripts.training import train_unified_clean as T

NAME = "fig_overview"
EVENT_ID = "901cb9ad-e904-553c-9ccb-08a45f9d5895"   # Real Sociedad 的一次完成传球，24/25 干净场次 3946608，定格帧可见 20 人
# Raw StatsBomb directory (see scripts/cache/prepare_l1_events.py); only needed to redraw the example frame.
DATOS = Path(os.environ.get("STATSBOMB_DATA_DIR", "data/statsbomb"))
SEED_PT = DEV / "data" / "cache" / "u1_clean" / "u1_s3.pt"
SEED_NPZ = DEV / "data" / "cache" / "u1_clean" / "u1_s3.npz"

# 原图的配色（从 fig1.pdf 渲染图取色）
OR, OR_L = "#E97132", "#F2AA84"          # 标量分支
BL, BL_L, BL_T = "#0B76A0", "#61CBF4", "#4E95D9"   # 空间分支
PU, PU_L, PU_B = "#7030A0", "#D86ECC", "#A893C2"   # 预测头和门控
NAVY, RED = "#1F4E79", "#CC3311"
GRAY_CELL = "#D9D9D9"
W, H = 17.5, 12.45   # 画布，厘米
FS = 7.5   # 全图统一字号（17.5 cm 宽，缩到 16.4 cm 版心后约 7 pt，是下限）


def arch():
    """从模型类读出图里要写的每个维数（和方框版同名同口径，verify_all 用它回读）"""
    n_scalar = len(T.F_BASE) + len(T.F_PASS) + len(T.F_SB)
    m = T.UnifiedGatingClean(event_dim=n_scalar, n_tasks=len(T.TASKS), fusion="gate", cnn_channels=7)
    enc = [l for l in m.event_enc if isinstance(l, nn.Linear)]
    conv = [l for l in m.spatial_enc if isinstance(l, nn.Conv2d)]
    pool = [l for l in m.spatial_enc if isinstance(l, nn.AdaptiveAvgPool2d)][0]
    lin_s = [l for l in m.spatial_enc if isinstance(l, nn.Linear)][0]
    gate = m.gate[0]
    hid = m.hidden[0]
    drop = [l for l in m.event_enc if isinstance(l, nn.Dropout)][0]
    a = dict(n_scalar=n_scalar, n_rich=15, task_dim=m.task_embed.embedding_dim, n_tasks=m.task_embed.num_embeddings,
             enc_in=enc[0].in_features, enc_h=enc[0].out_features, enc_out=enc[1].out_features,
             act_e=[type(l).__name__ for l in m.event_enc], act_s=[type(l).__name__ for l in m.spatial_enc],
             c_in=conv[0].in_channels, c1=conv[0].out_channels, k1=conv[0].kernel_size[0], c2=conv[1].out_channels, k2=conv[1].kernel_size[0],
             pool=tuple(np.atleast_1d(pool.output_size)) if not isinstance(pool.output_size, tuple) else pool.output_size,
             lin_s_in=lin_s.in_features, lin_s_out=lin_s.out_features, gate_in=gate.in_features, gate_out=gate.out_features,
             hid_in=hid.in_features, hid_out=hid.out_features, out_bin=m.out_bin.out_features, out_dest=m.out_dest.out_features,
             n_dest=T.N_DEST, grid=(8, 12), dropout=drop.p)
    assert a["enc_in"] == a["n_scalar"] + a["task_dim"]
    assert a["gate_in"] == a["enc_out"] + a["lin_s_out"] and a["gate_out"] == a["enc_out"] == a["lin_s_out"]
    assert isinstance(m.gate[1], nn.Sigmoid)
    assert a["act_e"] == ["Linear", "ELU", "Dropout", "Linear", "Tanh"], a["act_e"]
    assert a["act_s"] == ["Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d", "Conv2d", "BatchNorm2d", "ReLU", "AdaptiveAvgPool2d", "Flatten", "Linear", "Tanh"]
    assert [type(l).__name__ for l in m.hidden] == ["Linear", "ELU", "Dropout"] and m.hidden[2].p == a["dropout"]
    return a


def event_data():
    """同一次传球事件的定格帧、张量和 64 维门控"""
    cols = list(dict.fromkeys(["event_id", "match_id", "season_dir", "player_name", "team_name", "pass_outcome_name"] + T.F_BASE + T.F_PASS + T.F_SB))
    df = pd.read_parquet(DEV / "data" / "cache" / "L1_events_v3.parquet", columns=cols)
    r = df[df["event_id"] == EVENT_ID].copy()
    assert len(r) == 1
    st = pd.read_parquet(DEV / "audit" / "raw_scan" / "match_state_2425.parquet")
    clean = set(st.loc[(st["state"] == "正常") & (st["shot_frame_pct"] >= 70), "match_id"].astype(int))
    mid = int(r["match_id"].iloc[0])
    assert mid in clean, "示意图用的事件必须来自 202 场干净比赛"
    frames = json.load(open(DATOS / r["season_dir"].iloc[0] / f"{mid}_360.json", encoding="utf8"))
    ff = next(x for x in frames if x["event_uuid"] == EVENT_ID)["freeze_frame"]
    idx = pd.read_parquet(DEV / "data" / "cache" / "action_soccermaps_pass_idx.parquet")
    row = int(np.where(idx["event_id"].values == EVENT_ID)[0][0])
    sm = np.asarray(np.load(DEV / "data" / "cache" / "action_soccermaps_pass.npy", mmap_mode="r")[row], dtype=np.float32)
    # 门控  落点任务，特征照 build_dataset 的 feats（落点任务不给传球特征），标准化用权重文件里存的训练时参数
    r["under_pressure"] = r["under_pressure"].fillna(False).astype(int)
    for c in ["pass_is_progressive", "pass_cross", "pass_switch", "pass_through_ball", "pass_cut_back"]:
        r[c] = r[c].fillna(False).astype(int)
    X = np.hstack([r[T.F_BASE].fillna(0).values, np.zeros((1, len(T.F_PASS))), r[T.F_SB].fillna(0).values]).astype(np.float32)
    assert X.shape == (1, len(T.F_BASE) + len(T.F_PASS) + len(T.F_SB))
    ck = torch.load(SEED_PT, map_location="cpu", weights_only=False)
    m = T.UnifiedGatingClean(event_dim=X.shape[1], n_tasks=len(T.TASKS))
    m.load_state_dict(ck["state"])
    m.eval()
    with torch.no_grad():
        _, _, g = m(torch.tensor((X - ck["scaler_mean"]) / ck["scaler_scale"], dtype=torch.float32), torch.tensor(sm[None]),
                    torch.tensor([T.TASK_ID["dest"]]))
    g = g.numpy()[0]
    z = np.load(SEED_NPZ, allow_pickle=True)
    k = np.where((z["event_id"] == EVENT_ID) & (z["task"] == T.TASK_ID["dest"]))[0]
    assert len(k) == 1 and abs(float(z["gate_mean"][k[0]]) - float(g.mean())) < 1e-6, "门控重算和训练时存的门控均值不一致"
    return r.iloc[0], ff, sm, g, float(z["gate_mean"][k[0]])


# ---------------------------------------------------------------- 画图小工具（坐标单位厘米）
FRAMES = []   # 画过的框 (x0, y0, x1, y1, 是否虚线)，check_layout 用来查文字压框


def rbox(ax, x0, y0, w, h, ec, lw=1.1, ls="-", fc="white", r=0.22, z=1, track=True):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", linewidth=lw, linestyle=ls,
                                edgecolor=ec, facecolor=fc, zorder=z))
    if track:
        FRAMES.append((x0, y0, x0 + w, y0 + h, ls != "-"))


def arrow(ax, p, q, lw=0.9, ms=7, z=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color="black", shrinkA=0, shrinkB=0, zorder=z))


def neuron_col(ax, xc, ytop, ybot, r, fc, ec, n_top=3, frame=True):
    """一列神经元  上面 n_top 个圆、竖排省略点、最下一个圆，外面套圆角框。返回各圆心"""
    ys = list(np.linspace(ytop - r - 0.06, ybot + r + 0.06, n_top + 2))
    centers = ys[:n_top] + [ys[-1]]
    if frame:
        rbox(ax, xc - r - 0.06, ybot, 2 * r + 0.12, ytop - ybot, ec, lw=0.9, r=0.08, z=2, track=False)
    for y in centers:
        ax.add_patch(Circle((xc, y), r, facecolor=fc, edgecolor=ec, linewidth=0.8, zorder=3))
    ymid = (ys[n_top - 1] + ys[-1]) / 2
    for dy in (-0.1, 0, 0.1):
        ax.add_patch(Circle((xc, ymid + dy), 0.02, facecolor="black", edgecolor="none", zorder=3))
    return [(xc, y) for y in centers]


def neuron_row(ax, xl, xr, yc, r, fc, ec, n_left=3):
    xs = list(np.linspace(xl + r + 0.06, xr - r - 0.06, n_left + 2))
    centers = xs[:n_left] + [xs[-1]]
    rbox(ax, xl, yc - r - 0.06, xr - xl, 2 * r + 0.12, ec, lw=0.9, r=0.08, z=2, track=False)
    for x in centers:
        ax.add_patch(Circle((x, yc), r, facecolor=fc, edgecolor=ec, linewidth=0.8, zorder=3))
    xmid = (xs[n_left - 1] + xs[-1]) / 2
    for dx in (-0.08, 0, 0.08):
        ax.add_patch(Circle((xmid + dx, yc), 0.02, facecolor="black", edgecolor="none", zorder=3))


def connect(ax, A, B, lw=0.5):
    for a in A:
        for b in B:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=lw, zorder=2.5)


def cell_strip(ax, x0, y0, w, h, n=7, dots_at=3):
    cw = w / n
    for i in range(n):
        fc = "white" if i == dots_at else GRAY_CELL
        ax.add_patch(Rectangle((x0 + i * cw, y0), cw, h, facecolor=fc, edgecolor="#404040", linewidth=0.6, zorder=3))
    for dx in (-0.09, 0, 0.09):
        ax.add_patch(Circle((x0 + (dots_at + 0.5) * cw + dx, y0 + h / 2), 0.027, facecolor="black", zorder=4))


def cube_stack(ax, x0, y0, w, h, nx, ny, k, dx, dy, fc=BL_T, ec=BL):
    """k 层错开叠放的网格，最前一层画格线"""
    for i in range(k)[::-1]:
        ox, oy = dx * i, dy * i
        ax.add_patch(Rectangle((x0 + ox, y0 + oy), w, h, facecolor=fc, edgecolor=ec, linewidth=0.5, zorder=3 + (k - i) * 0.01))
    for i in range(1, nx):
        ax.plot([x0 + i * w / nx] * 2, [y0, y0 + h], color=ec, lw=0.35, zorder=3.2)
    for j in range(1, ny):
        ax.plot([x0, x0 + w], [y0 + j * h / ny] * 2, color=ec, lw=0.35, zorder=3.2)


def txt(ax, x, y, s, fs=FS, ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=fs, ha=ha, va=va, zorder=8, **kw)


def check_layout(fig, ax, margin_dash=0.14, margin_solid=0.06):
    """查每段文字有没有压框（压着或贴着框线，虚线框留半个字高），文字之间有没有重叠。返回问题清单"""
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    boxes = []
    for t in ax.texts:
        if not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        boxes.append((t.get_text().replace("\n", " ")[:30], x0, y0, x1, y1))
    probs = []
    for name, x0, y0, x1, y1 in boxes:
        for fx0, fy0, fx1, fy1, dashed in FRAMES:
            m = margin_dash if dashed else margin_solid
            touch = x1 > fx0 - m and x0 < fx1 + m and y1 > fy0 - m and y0 < fy1 + m
            inside = x0 >= fx0 + m and x1 <= fx1 - m and y0 >= fy0 + m and y1 <= fy1 - m
            if touch and not inside:
                probs.append(f"文字压框  {name!r}  框 ({fx0:.2f},{fy0:.2f})-({fx1:.2f},{fy1:.2f}) {'虚线' if dashed else '实线'}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[3] > b[1] and a[1] < b[3] and a[4] > b[2] and a[2] < b[4]:
                probs.append(f"文字重叠  {a[0]!r} 和 {b[0]!r}")
    return probs


def main():
    R = Registry(NAME)
    a = arch()
    for k, v in a.items():
        R.add(f"arch/{k}", v if isinstance(v, (int, float)) else str(v),
              dict(kind="code", file="scripts/training/train_unified_clean.py", where="UnifiedGatingClean 实例的层属性"))
    ev, ff, sm, g, g_npz = event_data()
    R.notes.append(f"示意图事件 {EVENT_ID}，{ev['team_name']} {ev['player_name']} 在 ({ev['location_x']}, {ev['location_y']}) 的完成传球，"
                   f"定格帧可见 {len(ff)} 人；u1 种子 3 落点任务 64 维门控均值重算 {g.mean():.6f}，u1_s3.npz 存的 {g_npz:.6f}")
    (OUT_DIR / f"{NAME}_event.json").write_text(json.dumps(dict(event_id=EVENT_ID, team=ev["team_name"], player=ev["player_name"],
                                                                 x=float(ev["location_x"]), y=float(ev["location_y"]), n_visible=len(ff),
                                                                 gate=[round(float(v), 6) for v in g], gate_mean=float(g.mean()),
                                                                 gate_mean_npz=g_npz), ensure_ascii=False, indent=1), encoding="utf8")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "savefig.dpi": 300})
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    FRAMES.clear()
    TOP = 12.3   # 四个区块标题顶端对齐

    def inset(x0, y0, w, h):
        return fig.add_axes([x0 / W, y0 / H, w / W, h / H])

    # ============ (1) INPUTS（x 0.15 到 3.55）
    txt(ax, 1.85, TOP, "(1) INPUTS", weight="bold", va="top")
    txt(ax, 1.85, 11.93, "from a StatsBomb 360\nfreeze-frame", va="top", linespacing=1.12)
    from mplsoccer import Pitch
    pax = inset(0.45, 9.6, 2.8, 1.87)
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#B7D7B0", line_color="white", stripe=True, stripe_color="#A6CCA0",
                  linewidth=0.6, pad_left=1, pad_right=1, pad_top=1, pad_bottom=1)
    pitch.draw(ax=pax)
    for p in ff:
        if p.get("actor"):
            continue
        x, y = p["location"]
        pitch.scatter(x, y, ax=pax, s=8, color=RED if p["teammate"] else NAVY, edgecolors="white", linewidth=0.3, zorder=3)
    act = next(p for p in ff if p.get("actor"))
    pitch.scatter(act["location"][0], act["location"][1], ax=pax, s=20, marker="s", color=RED, edgecolors="black", linewidth=0.5, zorder=4)
    pitch.scatter(float(ev["location_x"]), float(ev["location_y"]), ax=pax, s=7, color="white", edgecolors="black", linewidth=0.5, zorder=5)
    ax.plot([0.24, 0.24], [9.6, 4.45], color="black", lw=0.9, zorder=5)
    for yy in (8.65, 7.15, 4.45):
        arrow(ax, (0.24, yy), (0.42, yy))
    rbox(ax, 0.42, 8.0, 3.13, 1.3, OR)
    txt(ax, 0.56, 9.06, r"Event scalar features $x_{event}$", ha="left")
    cell_strip(ax, 0.64, 8.5, 2.7, 0.27)
    txt(ax, 1.99, 8.24, "location, action ...")
    rbox(ax, 0.42, 6.5, 3.13, 1.3, OR)
    txt(ax, 0.56, 7.56, r"360 scalar features $x_{360}$", ha="left")
    cell_strip(ax, 0.64, 7.0, 2.7, 0.27)
    txt(ax, 1.99, 6.74, "nearest ..., teammate ...")
    rbox(ax, 0.42, 3.3, 2.88, 2.65, BL)
    txt(ax, 1.86, 5.68, r"Spatial tensor $X_{spatial}$")
    txt(ax, 1.86, 5.36, f"({a['c_in']} x {a['grid'][0]} x {a['grid'][1]})")
    cube_stack(ax, 0.9, 3.58, 1.5, 1.0, a["grid"][1], a["grid"][0], 7, 0.06, 0.05)
    txt(ax, 1.86, 2.92, "7 channels on a\ncoarse pitch grid", linespacing=1.12)
    arrow(ax, (1.86, 2.57), (1.86, 2.24))

    # ============ (2) TWO PARALLEL ENCODERS（x 3.65 到 9.1）
    txt(ax, 6.35, TOP, "(2) TWO PARALLEL ENCODERS", weight="bold", va="top")
    rbox(ax, 4.05, 10.42, 4.35, 1.5, OR, lw=1.2)
    txt(ax, 6.22, 11.7, "Concatenate")
    txt(ax, 6.22, 11.42, r"$[x_{event};\ x_{360};\ e_t]$")
    cell_strip(ax, 4.85, 10.98, 2.75, 0.24)
    txt(ax, 6.22, 10.66, r"task embedding $e_t \in \mathbb{R}^{%d}$ (unified model)" % a["task_dim"])
    ax.plot([3.55, 3.72, 3.72], [8.65, 8.65, 11.1], color="black", lw=0.9, zorder=5)
    ax.plot([3.55, 3.72], [7.15, 7.15], color="black", lw=0.9, zorder=5)
    ax.plot([3.72, 3.72], [7.15, 8.65], color="black", lw=0.9, zorder=5)
    arrow(ax, (3.72, 11.1), (4.05, 11.1))
    arrow(ax, (6.22, 10.42), (6.22, 10.1))
    rbox(ax, 3.88, 6.15, 4.3, 3.95, OR, lw=1.2, r=0.45)
    txt(ax, 6.03, 9.83, r"$\phi_{event}$, 2-layer MLP")
    txt(ax, 6.03, 9.3, "Linear → ELU → Dropout(%.1f)\n→ Linear → Tanh" % a["dropout"], linespacing=1.2)
    c1 = neuron_col(ax, 4.85, 8.5, 6.4, 0.18, OR_L, OR)
    c2 = neuron_col(ax, 6.03, 8.7, 6.3, 0.18, OR_L, OR, n_top=4)
    c3 = neuron_col(ax, 7.21, 8.5, 6.4, 0.18, OR_L, OR)
    connect(ax, c1, c2)
    connect(ax, c2, c3)
    arrow(ax, (8.18, 7.55), (8.62, 7.55))
    txt(ax, 8.6, 9.72, r"$h_e \in \mathbb{R}^{%d}$" % a["enc_out"])
    neuron_col(ax, 8.86, 9.45, 6.4, 0.2, OR_L, OR, n_top=4)
    # CNN 路径  每个小立方体上方两三行说明，维数写在立方体下方
    arrow(ax, (3.3, 4.45), (3.45, 4.45))
    rbox(ax, 3.45, 2.75, 3.9, 2.4, BL, lw=1.2, r=0.4)
    txt(ax, 3.58, 4.98, "Conv %d×%d, %d→%d\n+ BN + ReLU\n+ MaxPool$_2$" % (a["k1"], a["k1"], a["c_in"], a["c1"]), ha="left", va="top",
        linespacing=1.12)
    txt(ax, 5.43, 4.98, "Conv %d×%d, %d→%d\n+ BN + ReLU\n+ AvgPool$_{%d\\times%d}$" % (a["k2"], a["k2"], a["c1"], a["c2"], a["pool"][0], a["pool"][1]),
        ha="left", va="top", linespacing=1.12)
    cube_stack(ax, 3.72, 3.3, 0.8, 0.5, 6, 4, 6, 0.045, 0.04)
    txt(ax, 4.88, 3.4, str(a["c1"]))
    txt(ax, 4.1, 3.0, "%d×%d×%d" % (a["c1"], a["grid"][0] // 2, a["grid"][1] // 2))
    arrow(ax, (5.05, 3.58), (5.5, 3.58))
    cube_stack(ax, 5.68, 3.4, 0.32, 0.32, 2, 2, 6, 0.04, 0.035)
    txt(ax, 6.45, 3.47, str(a["c2"]))
    txt(ax, 5.98, 3.0, "%d×%d×%d" % (a["c2"], a["pool"][0], a["pool"][1]))
    arrow(ax, (7.35, 3.85), (7.47, 3.85))
    txt(ax, 7.6, 5.35, "flatten\n(%d)" % a["lin_s_in"], linespacing=1.08)
    neuron_col(ax, 7.6, 4.8, 2.9, 0.12, BL_L, BL)
    arrow(ax, (7.78, 3.85), (8.12, 3.85))
    txt(ax, 8.3, 5.35, "Linear\n+ Tanh", linespacing=1.08)
    neuron_col(ax, 8.3, 4.8, 2.9, 0.12, BL_L, BL)
    arrow(ax, (8.48, 3.85), (8.66, 3.85))
    neuron_col(ax, 8.86, 5.75, 2.75, 0.2, BL_L, BL, n_top=4)
    txt(ax, 8.86, 2.47, r"$h_s \in \mathbb{R}^{%d}$" % a["lin_s_out"])

    # ============ (3) TASK-ADAPTIVE SPATIAL GATING（x 9.15 到 13.02）
    rbox(ax, 9.15, 3.45, 3.87, 8.93, "black", lw=0.9, ls=(0, (3, 2)), r=0.4, z=0.5)
    txt(ax, 11.08, TOP - 0.18, "(3) TASK-ADAPTIVE\nSPATIAL GATING", weight="bold", va="top", linespacing=1.12)
    txt(ax, 9.98, 11.25, r"$h_e \in \mathbb{R}^{%d}$" % a["enc_out"])
    txt(ax, 12.18, 11.25, r"$h_s \in \mathbb{R}^{%d}$" % a["lin_s_out"])
    neuron_col(ax, 9.98, 11.0, 9.6, 0.13, OR_L, OR)
    neuron_col(ax, 12.18, 11.0, 9.6, 0.13, BL_L, BL)
    arrow(ax, (9.98, 9.56), (9.98, 9.26))
    arrow(ax, (12.18, 9.56), (12.18, 9.26))
    rbox(ax, 9.42, 8.2, 3.32, 1.03, "#BFBFBF", lw=1.0, r=0.18)
    txt(ax, 11.08, 8.9, r"$g = \sigma\,(W_g[h_e;\,h_s] + b_g),$")
    txt(ax, 11.08, 8.52, r"$g \in [0,1]^{%d}$" % a["gate_out"])
    gax = inset(10.05, 6.8, 2.6, 0.92)
    gax.bar(np.arange(len(g)), g, width=0.8, color=PU_B, edgecolor="none")
    gax.set_xlim(-1, len(g))
    gax.set_ylim(0, 1.05)
    gax.set_yticks([0, 0.5, 1.0])
    gax.set_yticklabels(["0", "0.5", "1"])
    gax.set_xticks([])
    for sp in ("top", "right"):
        gax.spines[sp].set_visible(False)
    gax.spines["left"].set_linewidth(0.6)
    gax.spines["bottom"].set_linewidth(0.6)
    gax.tick_params(labelsize=FS, length=2, width=0.6, pad=1)
    txt(ax, 11.08, 7.96, r"Gate values $g_d$ of one pass")
    txt(ax, 11.35, 6.6, "64 gate dimensions")
    ax.plot([9.42, 12.74], [6.35, 6.35], color="#BFBFBF", lw=0.8, ls=(0, (3, 2)), zorder=1)
    txt(ax, 11.08, 6.1, "Fusion (element-wise)")
    rbox(ax, 9.42, 5.38, 3.32, 0.46, "#BFBFBF", lw=1.0, r=0.12)
    txt(ax, 11.08, 5.61, r"$h_{fused} = g \odot h_s + (1-g) \odot h_e$")
    txt(ax, 11.08, 5.0, r"$g$, gate value per dimension")
    txt(ax, 11.08, 4.66, r"$h_s$, spatial embedding")
    txt(ax, 11.08, 4.32, r"$h_e$, event embedding")
    arrow(ax, (11.4, 3.45), (11.4, 2.8))
    txt(ax, 11.28, 3.12, r"$h_{fused} \in \mathbb{R}^{%d}$" % a["gate_out"], ha="right")
    neuron_row(ax, 10.65, 12.15, 2.6, 0.11, PU_L, PU)
    arrow(ax, (12.15, 2.6), (13.1, 2.6))

    # ============ (4) PREDICTION HEAD（x 13.1 到 15.92）
    rbox(ax, 13.1, 2.3, 2.82, 10.08, "black", lw=0.9, ls=(0, (3, 2)), r=0.4, z=0.5)
    txt(ax, 14.51, TOP - 0.18, "(4) PREDICTION HEAD", weight="bold", va="top")
    rbox(ax, 13.27, 9.25, 2.48, 2.45, PU, lw=1.2, r=0.3)
    txt(ax, 14.51, 11.45, "Linear %d → %d" % (a["hid_in"], a["hid_out"]), weight="bold")
    h1 = neuron_col(ax, 14.0, 11.15, 9.85, 0.12, PU_L, PU)
    h2 = neuron_col(ax, 15.02, 11.15, 9.85, 0.12, PU_L, PU)
    connect(ax, h1, h2, lw=0.45)
    txt(ax, 14.51, 9.52, "ELU → Dropout(%.1f)" % a["dropout"])
    for xc, nout, l1, l2 in ((13.88, a["out_bin"], "logit", "(real value)"), (15.14, a["out_dest"], "%d logits" % a["out_dest"], "(per cell)")):
        arrow(ax, (xc, 9.25), (xc, 8.95))
        rbox(ax, xc - 0.56, 7.0, 1.12, 1.95, PU, lw=1.2, r=0.2)
        txt(ax, xc, 8.6, "Linear\n%d → %d" % (a["hid_out"], nout), weight="bold", linespacing=1.08)
        neuron_col(ax, xc, 8.15, 7.15, 0.11, PU_L, PU, n_top=2)
        arrow(ax, (xc, 7.0), (xc, 6.75))
        txt(ax, xc, 6.55, l1, weight="bold")
        txt(ax, xc, 6.25, l2)
        ax.plot([xc, xc], [6.05, 5.78], color=PU, lw=0.9, ls=(0, (2, 1.5)), zorder=5)
    rbox(ax, 13.24, 2.45, 2.54, 3.33, PU, lw=1.1, ls=(0, (3, 2)), r=0.3)
    txt(ax, 14.51, 5.48, "Loss head", weight="bold")
    for xc, label in ((13.88, "BCE\nfor the\nseven\noutcome\ntasks"), (15.14, "Softmax\nover %d\ncells for\npass\ndestination" % a["out_dest"])):
        rbox(ax, xc - 0.6, 2.62, 1.2, 2.55, PU, lw=1.1, r=0.15)
        txt(ax, xc, 3.9, label, linespacing=1.12)

    # ============ (5) Output
    arrow(ax, (15.92, 7.9), (16.12, 7.9))
    txt(ax, 16.8, 8.15, "(5) Output", weight="bold")
    txt(ax, 16.8, 7.68, r"$P(y\,|\,event)$", fs=FS + 1.5)

    # ============ 底部七个通道（y 0.05 到 2.2）
    rbox(ax, 0.12, 0.05, 17.26, 2.13, "black", lw=1.0, r=0.3, z=0.5)
    names = ["A. Teammates", "B. Opponents", "C. Ball dist.", "D. Goal dist.", r"E. sin $\theta$ ball", r"F. cos $\theta$ ball", r"G. $\theta$ to goal"]
    x0, wpan, hpan, gap = 0.35, 2.04, 1.36, 0.17
    for c in range(7):
        v = sm[c]
        vn = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.zeros_like(v)
        hx = inset(x0 + c * (wpan + gap), 0.22, wpan, hpan)
        hp = Pitch(pitch_type="statsbomb", line_color="white", pitch_color="none", linewidth=0.5, pad_left=0, pad_right=0, pad_top=0,
                   pad_bottom=0, line_zorder=2)
        hp.draw(ax=hx)
        hx.imshow(vn, extent=(0, 120, 80, 0), origin="upper", cmap="cividis", vmin=0, vmax=1, zorder=1, interpolation="nearest", aspect="auto")
        txt(ax, x0 + c * (wpan + gap) + wpan / 2, 1.9, names[c], weight="bold")
    cax = inset(15.9, 0.3, 0.15, 1.25)
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap="cividis", norm=plt.Normalize(0, 1)), cax=cax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["0", "0.5", "1"])
    cb.ax.tick_params(labelsize=FS, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)
    cb.set_label("normalized intensity", fontsize=FS, labelpad=2, rotation=90)
    probs = check_layout(fig, ax)
    for pr in probs:
        print("[layout]", pr)
    R.notes.append(f"版式自动检查  {len(probs)} 处文字压框或重叠" + (f"  {probs}" if probs else ""))
    save_fig(fig, NAME)
    R.save(extra=dict(arch=a))
    print(f"[{NAME}] 画好", flush=True)


if __name__ == "__main__":
    main()
