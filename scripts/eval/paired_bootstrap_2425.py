#!/usr/bin/env python3
"""
表 7 的 2024/25 配对自助区间  在 202 场干净测试比赛上按场次重抽样，给“模型减加厚 XGBoost”的种子均值差算 95% 区间
============================
做法照搬 scripts/eval/holdout_2526_summarize.py 的 bootstrap（AUC 用同一个 AUCFast，按场次整数权重重算；
所有任务、所有模型共用同一次重抽样；RandomState 20261003；2.5 和 97.5 分位）。差在种子均值上做，配对规则也照搬它的 paired_diff，
单任务模型之间按共有种子配对（同种子是同一个划分），统一模型对单任务模型是种子均值之差、不配对（冻结方案 C3 的写法）。
  A  薄统一模型 u1（种子 0 到 3）减加厚 XGBoost（种子 0 到 4），不配对，和表 9 C3 同一个量
  B  加厚 CNN（种子 0 到 2）减加厚 XGBoost，按共有种子 0 到 2 配对，就是正文 5.2 引的残差
  B5 同 B，但加厚 XGBoost 取五个种子的均值，和表 7 两格均值相减的口径一样
  C  加厚统一模型 u1rich（种子 0 到 3）减加厚 XGBoost，不配对（卡外顺手算，正文 5.3 第二句用到）
点估计先和表 7 的出处登记（scripts/paper_v2/out/tab_outcome_main_sources.json）里每个模型的种子均值四位核对，对不上就停。
只读 data/，不写 data/。

输入  data/cache/u1_clean/{u1,u1rich}_s*.npz，data/cache/single_clean_rich/preds/{task}_{model}_s*.npz 和 {task}_test_meta.npz，
      audit/raw_scan/match_state_2425.parquet（经 summarize_u1_clean.clean_matches 取 202 场），scripts/paper_v2/out/tab_outcome_main_sources.json
输出  audit/2026-10-06_paired_bootstrap_2425.md，audit/2026-10-06_paired_bootstrap_2425.json

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/eval/paired_bootstrap_2425.py --n_boot 2000

Last modified 2026-10-06
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.eval.holdout_2526_summarize import BOOT_SEED, AUCFast
from scripts.eval.summarize_u1_clean import clean_matches

U1 = Path("data/cache/u1_clean")
RICH = Path("data/cache/single_clean_rich/preds")
TASKS8 = ["pass", "dest", "shot", "interception", "ball_recovery", "pressure", "dribble", "tackle"]
TASKS7 = ["pass", "shot", "interception", "ball_recovery", "pressure", "dribble", "tackle"]
MEAN3 = ["pass", "ball_recovery", "pressure"]
MODELS = {  # 名字: (来源, 文件里的模型名或标签, 种子)
    "u1": ("unified", "u1", [0, 1, 2, 3]),
    "u1rich": ("unified", "u1rich", [0, 1, 2, 3]),
    "xgbrich": ("single", "B2_XGB_360", [0, 1, 2, 3, 4]),
    "cnnrich": ("single", "M4_CNN_Full", [0, 1, 2]),
}
REG_ID = {"u1": "unified_u1", "u1rich": "unified_u1rich", "xgbrich": "single_B2_XGB_360_rich", "cnnrich": "single_M4_CNN_Full_rich"}
# (键, 减数左边, 左边种子, 右边, 右边种子, 说明)
CONTRASTS = [
    ("A_u1_minus_xgbrich", "u1", [0, 1, 2, 3], "xgbrich", [0, 1, 2, 3, 4], "薄统一模型（四种子均值）减加厚 XGBoost（五种子均值），不配对"),
    ("B_cnnrich_minus_xgbrich_paired3", "cnnrich", [0, 1, 2], "xgbrich", [0, 1, 2], "加厚 CNN 减加厚 XGBoost，按种子 0 到 2 配对，等于两边三种子均值之差"),
    ("B5_cnnrich_minus_xgbrich_xgb5", "cnnrich", [0, 1, 2], "xgbrich", [0, 1, 2, 3, 4], "加厚 CNN（三种子均值）减加厚 XGBoost（五种子均值），表 7 两格相减的口径"),
    ("C_u1rich_minus_xgbrich", "u1rich", [0, 1, 2, 3], "xgbrich", [0, 1, 2, 3, 4], "加厚统一模型（四种子均值）减加厚 XGBoost（五种子均值），不配对，卡外"),
]


def load_runs():
    """返回 {(模型名, 种子, 任务): dict(score, y, match)}，都限制在 202 场干净比赛"""
    clean = clean_matches()
    runs = {}
    for name, (src, tag, seeds) in MODELS.items():
        for sd in seeds:
            if src == "unified":
                z = np.load(U1 / f"{tag}_s{sd}.npz", allow_pickle=True)
                ok = np.array([int(m) in clean for m in z["match_id"]])
                for t in TASKS7:
                    m = (z["task"] == TASKS8.index(t)) & ok
                    runs[(name, sd, t)] = dict(score=z["logit_bin"][m].astype(np.float64), y=z["y"][m].astype(int), match=z["match_id"][m].astype(np.int64))
            else:
                for t in TASKS7:
                    meta = np.load(RICH / f"{t}_test_meta.npz", allow_pickle=True)
                    ok = np.array([int(m) in clean for m in meta["match_id"]])
                    z = np.load(RICH / f"{t}_{tag}_s{sd}.npz", allow_pickle=True)
                    runs[(name, sd, t)] = dict(score=z["prob"][ok].astype(np.float64), y=meta["y"][ok].astype(int), match=meta["match_id"][ok].astype(np.int64))
    return runs


def seed_means(runs, w_of):
    """{(模型名, 任务): {种子: AUC}}，w_of(run) 给权重，None 表示不加权"""
    out = {}
    for (name, sd, t), r in runs.items():
        w = w_of(r)
        out.setdefault((name, t), {})[sd] = r["fast"].auc(np.ones(len(r["y"])) if w is None else w)
    return out


def contrasts(per):
    """每个对比在每个任务上的种子均值差，再加七任务和三任务（Pass、Ball recovery、Pressure）平均"""
    out = {}
    for key, a, sa, b, sb, _ in CONTRASTS:
        d = {t: float(np.mean([per[(a, t)][s] for s in sa]) - np.mean([per[(b, t)][s] for s in sb])) for t in TASKS7}
        d["mean7"] = float(np.mean([d[t] for t in TASKS7]))
        d["mean3"] = float(np.mean([d[t] for t in MEAN3]))
        out[key] = d
    return out


def per_seed_paired(per):
    """B 的逐种子差，用来复述正文“每个种子都为正”这类说法"""
    return {t: {s: per[("cnnrich", t)][s] - per[("xgbrich", t)][s] for s in [0, 1, 2]} for t in TASKS7}


def check_against_table7(per):
    reg = json.load(open("scripts/paper_v2/out/tab_outcome_main_sources.json", encoding="utf8"))
    val = {it["id"]: it["value"] for it in reg["items"]}
    lines = []
    for name in MODELS:
        for t in TASKS7:
            m = float(np.mean(list(per[(name, t)].values())))
            ref = val[f"{t}/{REG_ID[name]}/mean"]
            ok = round(m, 4) == round(ref, 4)
            lines.append((name, t, m, ref, ok))
            if not ok:
                raise SystemExit(f"[check] {name} {t} 重算 {m:.6f} 和表 7 登记 {ref} 四位对不上，停")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    runs = load_runs()
    matches = np.unique(np.concatenate([r["match"] for r in runs.values()]))
    idx = {int(m): i for i, m in enumerate(matches)}
    for r in runs.values():
        r["fast"] = AUCFast(r["score"], r["y"])
        r["_mi"] = np.array([idx[int(m)] for m in r["match"]])
    print(f"[load] {len(runs)} 个模型×种子×任务，{len(matches)} 场  ({time.time() - t0:.0f}s)", flush=True)
    assert len(matches) == 202, len(matches)
    n_ev = {t: len(runs[("xgbrich", 0, t)]["y"]) for t in TASKS7}
    # 统一模型和单任务模型的测试事件要是同一批，按场次比事件数和正例数
    for (name, sd, t), r in runs.items():
        ref = runs[("xgbrich", 0, t)]
        same_n = (np.bincount(r["_mi"], minlength=len(matches)) == np.bincount(ref["_mi"], minlength=len(matches))).all()
        same_pos = (np.bincount(r["_mi"], r["y"], minlength=len(matches)) == np.bincount(ref["_mi"], ref["y"], minlength=len(matches))).all()
        assert same_n and same_pos, f"{name} s{sd} {t} 逐场事件数或正例数和加厚 XGBoost 不同"

    per = seed_means(runs, lambda r: None)
    chk = check_against_table7(per)
    print(f"[check] 四个模型七个任务的种子均值和表 7 登记四位一致（{len(chk)} 项）", flush=True)
    point = contrasts(per)
    seeds_b = per_seed_paired(per)

    rng = np.random.RandomState(BOOT_SEED)
    boots = {k: {t: [] for t in TASKS7 + ["mean7", "mean3"]} for k, *_ in CONTRASTS}
    t1 = time.time()
    for b in range(a.n_boot):
        counts = np.bincount(rng.randint(0, len(matches), len(matches)), minlength=len(matches)).astype(np.float64)
        pb = seed_means(runs, lambda r: counts[r["_mi"]])
        for k, d in contrasts(pb).items():
            for t, v in d.items():
                boots[k][t].append(v)
        if (b + 1) % 200 == 0:
            el = time.time() - t1
            print(f"[boot] {b + 1}/{a.n_boot}  {el:.0f}s  预计还要 {el / (b + 1) * (a.n_boot - b - 1):.0f}s", flush=True)

    res = {}
    for k, *_ in CONTRASTS:
        res[k] = {}
        for t in TASKS7 + ["mean7", "mean3"]:
            v = np.array(boots[k][t])
            res[k][t] = dict(point=point[k][t], lo=float(np.nanpercentile(v, 2.5)), hi=float(np.nanpercentile(v, 97.5)),
                             share_pos=float((v > 0).mean()), n_nan=int(np.isnan(v).sum()))
    out = dict(when=time.strftime("%Y-%m-%d %H:%M"), n_boot=a.n_boot, boot_seed=BOOT_SEED, n_matches=int(len(matches)), n_events=n_ev,
               contrasts={k: dict(desc=desc, left=lft, left_seeds=sa, right=rgt, right_seeds=sb) for k, lft, sa, rgt, sb, desc in CONTRASTS},
               results=res, per_seed_B={t: {str(s): v for s, v in d.items()} for t, d in seeds_b.items()},
               seed_means={f"{n}|{t}": {str(s): v for s, v in d.items()} for (n, t), d in per.items()})
    Path("audit/2026-10-06_paired_bootstrap_2425.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8")
    for k, *_ in CONTRASTS:
        print(f"[res] {k}", flush=True)
        for t in TASKS7 + ["mean7", "mean3"]:
            r = res[k][t]
            print(f"      {t:14s} {r['point']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  正的重抽样占 {r['share_pos']:.3f}", flush=True)
    print(f"[done] audit/2026-10-06_paired_bootstrap_2425.json  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
