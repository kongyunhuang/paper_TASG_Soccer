#!/usr/bin/env python3
"""
传球落点任务的规则基线  不用任何模型，看 96 格落点预测靠简单规则能到多少
============================
数据直接取 train_unified_clean.build_dataset 的 dest 任务，所以事件、标签、测试集和统一模型、单任务模型逐条相同。
指标只在 24/25 的 202 场干净比赛上算，和 summarize_u1_clean.py 同口径。
  均匀分布            每格 1/96
  边际分布            训练赛季里落点格的频率
  起点格条件表        按传球起点所在格查落点分布（加 0.2 平滑）
  条件表乘队友占位    p 正比于 表 × (1 + a × 该格队友数)，a 在训练赛季里留出的验证比赛上按对数损失挑，不看测试集
  条件表乘队友再除对手  p 正比于 表 × (1 + a × 队友数) / (1 + b × 对手数)，a、b 同样在验证比赛上挑
模型要说“学到了规则之外的东西”，top-1 和对数损失都得明显好过最后两行。

输入  同 train_unified_clean.py，另读 audit/raw_scan/match_state_2425.parquet
输出  audit/dest_rule_baselines.md

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/eval/dest_rule_baselines.py

Last modified 2026-10-02
"""
import itertools
import math
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.eval.summarize_u1_clean import clean_matches
from scripts.training.train_unified_clean import N_DEST, build_dataset

OUT = Path("audit/dest_rule_baselines.md")
SMOOTH = 0.2
GRID_A = [0, 0.5, 1, 2, 3, 5, 10, 20]
GRID_B = [0, 0.5, 1, 2, 5]


def start_cell(X):
    """起点所在格，和标签同一套 8×12 网格（每格 10 码，行优先）"""
    return np.clip((X[:, 1] / 10).astype(int), 0, 7) * 12 + np.clip((X[:, 0] / 10).astype(int), 0, 11)


def table(start, y):
    T = np.full((N_DEST, N_DEST), SMOOTH)
    np.add.at(T, (start, y), 1)
    return T / T.sum(1, keepdims=True)


def rule(T, start, mates, opps, a, b):
    P = T[start] * (1 + a * mates) / (1 + b * opps)
    return P / P.sum(1, keepdims=True)


def scores(P, y):
    n = np.arange(len(y))
    return dict(top1=float((P.argmax(1) == y).mean()), top3=float((np.argsort(-P, 1)[:, :3] == y[:, None]).any(1).mean()),
                logloss=float(-np.log(P[n, y]).mean()))


def main():
    t0 = time.time()
    d = build_dataset()["dest"]
    dev, test = d["dev"], d["test"]
    clean = clean_matches()
    ok = np.array([int(m) in clean for m in test["match"]])
    Xt, yt, smt = test["X"][ok], test["y"][ok].astype(int), test["sm"][ok]
    yd = dev["y"].astype(int)
    sd, st = start_cell(dev["X"]), start_cell(Xt)
    flat = lambda sm, c: np.asarray(sm[:, c]).reshape(len(sm), -1)
    md, od, mt, ot = flat(dev["sm"], 0), flat(dev["sm"], 1), flat(smt, 0), flat(smt, 1)
    print(f"[data] dev {len(yd):,}  test (202 场) {len(yt):,}  ({time.time() - t0:.0f}s)", flush=True)

    # 在训练赛季里留出两成比赛挑 a、b
    matches = np.sort(np.unique(dev["match"]))
    perm = np.random.RandomState(20261002).permutation(len(matches))
    tr = np.isin(dev["match"], matches[perm[:int(0.8 * len(matches))]])
    va = ~tr
    Ttr = table(sd[tr], yd[tr])
    grid = []
    for a, b in itertools.product(GRID_A, GRID_B):
        grid.append((scores(rule(Ttr, sd[va], md[va], od[va], a, b), yd[va])["logloss"], a, b))
    best_a = min((g for g in grid if g[2] == 0), key=lambda g: g[0])
    best_ab = min(grid, key=lambda g: g[0])
    print(f"[select] 验证比赛上  只用队友 a={best_a[1]} logloss {best_a[0]:.4f}   队友加对手 a={best_ab[1]} b={best_ab[2]} logloss {best_ab[0]:.4f}", flush=True)

    T = table(sd, yd)
    pm = (np.bincount(yd, minlength=N_DEST) + 1.0)
    pm = pm / pm.sum()
    rows = []
    for name, P in [("均匀分布", np.full((len(yt), N_DEST), 1 / N_DEST)),
                    ("边际分布", np.tile(pm, (len(yt), 1))),
                    ("起点格条件表", T[st]),
                    (f"条件表乘队友占位（a={best_a[1]}）", rule(T, st, mt, ot, best_a[1], 0)),
                    (f"条件表乘队友再除对手（a={best_ab[1]}，b={best_ab[2]}）", rule(T, st, mt, ot, best_ab[1], best_ab[2]))]:
        s = scores(P, yt)
        rows.append(dict(rule=name, top1=round(s["top1"], 4), top3=round(s["top3"], 4), logloss=round(s["logloss"], 4),
                         skill=round(1 - s["logloss"] / math.log(N_DEST), 4)))
        print(f"[rule] {name:30s} top1 {s['top1']:.4f} top3 {s['top3']:.4f} logloss {s['logloss']:.4f}", flush=True)
    same = float((st == yt).mean())
    lines = ["# 传球落点任务的规则基线", "",
             f"由 `scripts/eval/dest_rule_baselines.py` 生成（{socket.gethostname()}）。数据和统一模型的 dest 任务逐条相同，训练赛季样本 {len(yd):,} 条，"
             f"测试只取 24/25 的 202 场干净比赛共 {len(yt):,} 条。a、b 在训练赛季留出的两成比赛上按对数损失挑，没有看测试集。", "",
             pd.DataFrame(rows).to_markdown(index=False), "",
             f"落点和起点在同一格的比例 {same:.4f}。skill 一列是 1 减去对数损失除以 ln96，和统一模型选模型用的量相同。", ""]
    OUT.write_text("\n".join(lines), encoding="utf8")
    print(f"[done] {(time.time() - t0) / 60:.1f} min  saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
