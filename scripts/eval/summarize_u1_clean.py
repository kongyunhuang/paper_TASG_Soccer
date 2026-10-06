#!/usr/bin/env python3
"""
统一模型干净版的结果汇总  门控排序、跨种子稳定性、和单任务 CNN 增益的对应、干预前后的门控变化
============================
读 train_unified_clean.py 的输出，按运行标签分组汇总。
  每个标签一张表  各任务的测试指标和门控均值，跨种子的均值、标准差、最小、最大，以及门控均值在各种子里的名次
  门控和增益      各任务门控均值（统一模型）对单任务实验量出的 CNN 增益（M4 减 M2，只在 229 场正常状态比赛上算）的 Spearman 相关，
                  逐种子算一遍，再用种子平均算一遍。传球落点的单任务增益目前只有第三视角会话试验六的一次结果，量纲是归一化对数损失
  干预            --shuffle_spatial 的运行和同种子的基准运行相比，被打乱任务的门控均值变了多少，其他任务变了多少
  融合对照        --fusion concat 的运行和同种子门控运行的各任务指标之差
  单任务模型      同数据单任务四个模型在 202 场上的成绩和同种子配对差
  门控代价        --gate_penalty 的运行，种子 0 的路径用来选 λ，种子 1 到 3 按预先写下的标准判定
  消融对照        有 ablation.json 的运行，门控均值对球员通道消融下降量的 Spearman，两通道版按预先标准判定

输入  data/cache/u1_clean/*.json
      data/cache/single_clean/*.jsonl（和统一模型同数据的单任务结果，有则优先用）
      data/cache/multiseed/ 和 data/cache/multiseed_fixed/ 的 jsonl 与 preds（旧来源，single_clean 没有时才用）
      data/cache/pilot_third_view/tasg_selection_gate_results.json（只取落点任务的单任务增益）
      audit/raw_scan/match_state_2425.parquet
输出  audit/u1_clean_summary.md

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/eval/summarize_u1_clean.py

Last modified 2026-10-02
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from scripts.training.train_all import CACHE_DIR

U1 = CACHE_DIR / "u1_clean"
OUT = Path("audit/u1_clean_summary.md")
TASKS = ["pass", "dest", "shot", "interception", "ball_recovery", "pressure", "dribble", "tackle"]


def clean_matches():
    """干净测试场次  状态正常且射门带帧率不低于 70% 的 24/25 比赛（202 场）。
    另外 27 场状态标为正常，但射门和出界传球的帧只在特定结果下才有（2026-10-02 独立核查发现），不能当测试集。"""
    st = pd.read_parquet("audit/raw_scan/match_state_2425.parquet")
    return set(st.loc[(st["state"] == "正常") & (st["shot_frame_pct"] >= 70), "match_id"].astype(int))


def recompute(r, clean):
    """用逐事件预测在干净场次上重算各任务的测试指标和门控均值，覆盖 json 里按 229 场算的数"""
    z = np.load(U1 / f"{r['tag']}_s{r['seed']}.npz", allow_pickle=True)
    ok = np.array([int(m) in clean for m in z["match_id"]])
    for i, t in enumerate(TASKS):
        m = (z["task"] == i) & ok
        y = z["y"][m].astype(int)
        if t == "dest":
            nll = float(-z["dest_logp_true"][m].mean())
            te = dict(logloss=nll, top1=float((z["dest_top1"][m] == y).mean()), skill=1 - nll / math.log(96))
        else:
            auc = float(roc_auc_score(y, z["logit_bin"][m]))
            te = dict(auc=auc, skill=auc)
        r["tasks"][t]["test"] = te
        r["tasks"][t]["n_test"] = int(m.sum())
        if r["fusion"] == "gate":
            r["tasks"][t]["gate_mean"] = float(np.nanmean(z["gate_mean"][m]))
    return r


def load_runs():
    clean = clean_matches()
    runs = []
    for f in sorted(glob.glob(str(U1 / "*.json"))):
        if f.endswith("_ablation.json"):
            continue
        r = json.load(open(f))
        runs.append(recompute(r, clean))
    return runs


def metric(r, task, part="test"):
    m = r["tasks"][task][part]
    return m["top1"] if task == "dest" else m["auc"]


def single_task_gains():
    """单任务实验的 CNN 增益，M4 减 M2，在正常状态场次上按同种子配对后取平均。返回 {任务: (增益, 种子数, 备注)}"""
    clean = clean_matches()
    normal = clean
    out = {}
    for d, tasks in [(CACHE_DIR / "multiseed", ["pass", "shot", "interception", "ball_recovery", "pressure"]),
                     (CACHE_DIR / "multiseed_fixed", ["dribble", "tackle"])]:
        for t in tasks:
            f = d / f"{t}.jsonl"
            if not f.exists():
                continue
            res = pd.DataFrame([json.loads(x) for x in open(f)])
            meta = np.load(d / "preds" / f"{t}_test_meta.npz", allow_pickle=True)
            y = meta["y"]
            ok = np.array([int(m) in normal for m in meta["match_id"]])
            seeds = sorted(set(res[res["model"] == "M4_CNN_Full"]["seed"]) & set(res[res["model"] == "M2_MLP_360"]["seed"]))
            g = []
            for s in seeds:
                a = np.load(d / "preds" / f"{t}_M4_CNN_Full_s{s}.npz")["prob"]
                b = np.load(d / "preds" / f"{t}_M2_MLP_360_s{s}.npz")["prob"]
                if y[ok].min() == y[ok].max():
                    continue
                g.append(roc_auc_score(y[ok], a[ok]) - roc_auc_score(y[ok], b[ok]))
            if g:
                note = "旧标签（possession_team 定义），待用 5 秒标签重训" if t == "pressure" else ""
                out[t] = (float(np.mean(g)), len(g), note)
    # 和统一模型同数据的单任务结果（train_single_clean.py）优先，有就覆盖上面的旧来源
    sc = CACHE_DIR / "single_clean"
    for t in TASKS:
        f = sc / f"{t}.jsonl"
        if not f.exists():
            continue
        res = pd.DataFrame([json.loads(x) for x in open(f)])
        meta = np.load(sc / "preds" / f"{t}_test_meta.npz", allow_pickle=True)
        okc = np.array([int(m) in clean for m in meta["match_id"]])
        yy = meta["y"].astype(int)

        def skill(model, seed):
            z = np.load(sc / "preds" / f"{t}_{model}_s{seed}.npz")
            if t == "dest":
                return 1 + float(z["logp_true"][okc].mean()) / math.log(96)
            return float(roc_auc_score(yy[okc], z["prob"][okc]))

        s4 = set(res[res["model"] == "M4_CNN_Full"]["seed"]) & set(res[res["model"] == "M2_MLP_360"]["seed"])
        seeds = sorted(s for s in s4 if (sc / "preds" / f"{t}_M4_CNN_Full_s{s}.npz").exists() and (sc / "preds" / f"{t}_M2_MLP_360_s{s}.npz").exists())
        m4 = {s: skill("M4_CNN_Full", s) for s in seeds}
        m2 = {s: skill("M2_MLP_360", s) for s in seeds}
        if seeds:
            note = "归一化对数损失之差，和 AUC 之差不可直接比大小，只用于排序" if t == "dest" else ""
            out[t] = (float(np.mean([m4[s] - m2[s] for s in seeds])), len(seeds), ("同数据单任务。" + note) if note else "同数据单任务")
    if "dest" in out:
        return out
    pj = CACHE_DIR / "pilot_third_view" / "tasg_selection_gate_results.json"
    if pj.exists():
        p = json.load(open(pj))["传球落点"]
        gain = (p["M2_标量MLP"]["logloss"] - p["M4_CNN拼接"]["logloss"]) / math.log(96)
        out["dest"] = (float(gain), 1, "试验六单次结果，量纲是归一化对数损失之差，和 AUC 之差不可直接比大小，只用于排序")
    return out


def loss_unit_need():
    """和门控代价同单位的需要量  单任务 M4 相对 M2 的每事件对数损失下降量（202 场，五个种子平均），落点乘 1/ln96。
    训练集的逐事件预测没存，这里用测试集近似。返回 {任务: 下降量}"""
    clean = clean_matches()
    sc = CACHE_DIR / "single_clean"
    out = {}
    for t in TASKS:
        meta = np.load(sc / "preds" / f"{t}_test_meta.npz", allow_pickle=True)
        okc = np.array([int(m) in clean for m in meta["match_id"]])
        y = meta["y"].astype(int)[okc]
        d = []
        for sd in range(5):
            fa, fb = sc / "preds" / f"{t}_M4_CNN_Full_s{sd}.npz", sc / "preds" / f"{t}_M2_MLP_360_s{sd}.npz"
            if not (fa.exists() and fb.exists()):
                continue
            a, b = np.load(fa), np.load(fb)
            if t == "dest":
                d.append(float((a["logp_true"][okc] - b["logp_true"][okc]).mean()) / math.log(96))
            else:
                ll = lambda pr: -(y * np.log(np.clip(pr, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - pr, 1e-7, 1))).mean()
                d.append(float(ll(b["prob"][okc]) - ll(a["prob"][okc])))
        if d:
            out[t] = float(np.mean(d))
    return out


TIERS = [["dest", "pressure"], ["interception", "tackle"], ["pass", "shot", "dribble", "ball_recovery"]]


def tier_check(g):
    """分层检验  g 是 {任务: 门控均值}。返回（三层均值是否依次递减，20 个跨层任务对里次序反了的对数）"""
    means = [float(np.mean([g[t] for t in tier])) for tier in TIERS]
    viol = sum(g[a] <= g[b] for i in range(3) for j in range(i + 1, 3) for a in TIERS[i] for b in TIERS[j])
    return means[0] > means[1] > means[2], int(viol)


def plateau(r):
    """被选中的那一轮和它之前三轮相比，七个结果类任务验证集门控平均的变化。没有逐轮记录时返回 None"""
    h = r.get("history", [])
    if not h or h[0].get("val_gate") is None:
        return None
    best = int(np.argmax([x["val_score"] for x in h]))
    if best < 3:
        return float("nan")
    avg = lambda i: float(np.mean([h[i]["val_gate"][t] for t in TASKS if t != "dest"]))
    return avg(best) - avg(best - 3)


def single_task_table(dirname="single_clean", title="同数据单任务模型（202 场，跨种子均值 ± 标准差）"):
    """同数据单任务模型在 202 场上的逐模型成绩和配对差。dirname 取 single_clean（标量薄）或 single_clean_rich（标量加厚）。返回 markdown 行列表"""
    clean = clean_matches()
    sc = CACHE_DIR / dirname
    models = ["B2_XGB_360", "M2_MLP_360", "M4_CNN_Full", "G1_Gating"]
    rows, diffs = [], []
    for t in TASKS:
        if not (sc / f"{t}.jsonl").exists():
            continue
        meta = np.load(sc / "preds" / f"{t}_test_meta.npz", allow_pickle=True)
        okc = np.array([int(m) in clean for m in meta["match_id"]])
        yy = meta["y"].astype(int)
        val = {}
        for m in models:
            for f in sorted(glob.glob(str(sc / "preds" / f"{t}_{m}_s*.npz"))):
                seed = int(f.rsplit("_s", 1)[1].split(".")[0])
                z = np.load(f)
                if t == "dest":
                    val[(m, seed)] = (1 + float(z["logp_true"][okc].mean()) / math.log(96), float((z["top1"][okc] == yy[okc]).mean()))
                else:
                    val[(m, seed)] = (float(roc_auc_score(yy[okc], z["prob"][okc])), np.nan)
        row = dict(task=t, metric="skill（top1）" if t == "dest" else "AUC", n_test=int(okc.sum()))
        for m in models:
            v = np.array([val[k][0] for k in sorted(val) if k[0] == m])
            if len(v):
                cell = f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}"
                if t == "dest":
                    cell += f"（{np.mean([val[k][1] for k in val if k[0] == m]):.4f}）"
                row[m] = cell + f" n={len(v)}"
        rows.append(row)
        for a, b in [("M4_CNN_Full", "M2_MLP_360"), ("G1_Gating", "M2_MLP_360"), ("G1_Gating", "M4_CNN_Full"), ("M4_CNN_Full", "B2_XGB_360")]:
            seeds = sorted(set(k[1] for k in val if k[0] == a) & set(k[1] for k in val if k[0] == b))
            if not seeds:
                continue
            d = np.array([val[(a, s_)][0] - val[(b, s_)][0] for s_ in seeds])
            diffs.append(dict(task=t, pair=f"{a} 减 {b}", mean=round(d.mean(), 4), sd=round(d.std(ddof=1), 4) if len(d) > 1 else np.nan,
                              min=round(d.min(), 4), max=round(d.max(), 4), seeds_positive=f"{int((d > 0).sum())}/{len(d)}"))
    if not rows:
        return []
    return [f"## {title}", "",
            "落点任务的数是 skill（1 减对数损失除以 ln96），括号里是 top-1。XGBoost 不做落点任务。", "",
            pd.DataFrame(rows).to_markdown(index=False), "", "### 同种子配对差", "", pd.DataFrame(diffs).to_markdown(index=False), ""]


def main():
    runs = load_runs()
    lines = ["# 统一模型干净版结果汇总", "", "由 `scripts/eval/summarize_u1_clean.py` 生成。测试指标和门控均值都从逐事件预测重新算，只取 24/25 里的 202 场干净比赛。"
             "（360 状态正常的 229 场里有 27 场射门和出界传球的定格帧只在特定结果下才有，已剔除；训练时的测试集仍是 229 场，这里在算分时过滤。）", ""]
    if not runs:
        OUT.write_text("\n".join(lines + ["还没有结果。"]), encoding="utf8")
        print("no runs")
        return
    gains = single_task_gains()
    tags = sorted(set(r["tag"] for r in runs))
    by = {tag: sorted([r for r in runs if r["tag"] == tag], key=lambda r: r["seed"]) for tag in tags}

    for tag in tags:
        rs = by[tag]
        r0 = rs[0]
        lines += [f"## 运行 {tag}（融合 {r0['fusion']}，编码器 {r0.get('encoder', 'pooled')}，打乱的任务 {','.join(r0['shuffled']) or '无'}，种子 {[r['seed'] for r in rs]}）", ""]
        rows = []
        for t in TASKS:
            m = np.array([metric(r, t) for r in rs])
            row = dict(task=t, metric="top1" if t == "dest" else "AUC", test_mean=round(m.mean(), 4),
                       test_sd=round(m.std(ddof=1), 4) if len(m) > 1 else np.nan)
            if t == "dest":
                # 落点另给 skill（1 减对数损失除以 ln96）和对数损失，和规则基线、单任务表同口径
                sk = np.array([r["tasks"][t]["test"]["skill"] for r in rs])
                ll = np.array([r["tasks"][t]["test"]["logloss"] for r in rs])
                row.update(dest_skill_mean=round(sk.mean(), 4), dest_logloss_mean=round(ll.mean(), 4))
            if r0["fusion"] == "gate":
                g = np.array([r["tasks"][t]["gate_mean"] for r in rs])
                ranks = [int(sorted(TASKS, key=lambda x: -r["tasks"][x]["gate_mean"]).index(t)) + 1 for r in rs]
                row.update(gate_mean=round(g.mean(), 4), gate_sd=round(g.std(ddof=1), 4) if len(g) > 1 else np.nan,
                           gate_min=round(g.min(), 4), gate_max=round(g.max(), 4), gate_rank_by_seed=" ".join(map(str, ranks)))
            if t in gains:
                row.update(single_task_cnn_gain=round(gains[t][0], 4), gain_seeds=gains[t][1])
            rows.append(row)
        lines += [pd.DataFrame(rows).to_markdown(index=False), ""]
        if r0["fusion"] == "gate":
            common = [t for t in TASKS if t in gains]
            auc_only = [t for t in common if t != "dest"]
            for name, ts in [("含落点", common), ("只含 AUC 任务", auc_only)]:
                if len(ts) < 4:
                    continue
                per = []
                for r in rs:
                    rho, _ = spearmanr([r["tasks"][t]["gate_mean"] for t in ts], [gains[t][0] for t in ts])
                    per.append(rho)
                gm = [np.mean([r["tasks"][t]["gate_mean"] for r in rs]) for t in ts]
                rho, p = spearmanr(gm, [gains[t][0] for t in ts])
                lines += [f"门控均值对单任务 CNN 增益的 Spearman 相关（{name}，{len(ts)} 个任务）  逐种子 {', '.join(f'{x:.2f}' for x in per)}；种子平均后 {rho:.2f}（p {p:.3f}）", ""]
            notes = [f"{t} {gains[t][2]}" for t in common if gains[t][2]]
            if notes:
                lines += ["增益一列的说明  " + "；".join(notes), ""]

    # 门控均值对球员通道消融下降量（ablate_u1_saved.py 在 202 场上算的），逐种子的 Spearman 和 Pressure 的门控名次
    abl_rows = []
    for r in runs:
        f = U1 / f"{r['tag']}_s{r['seed']}_ablation.json"
        if r["fusion"] != "gate" or r["shuffled"] or not f.exists():
            continue
        ab = json.load(open(f))
        g = [r["tasks"][t]["gate_mean"] for t in TASKS]
        d = [ab[t]["drop_players"] for t in TASKS]
        rho, _ = spearmanr(g, d)
        rank_p = int(sorted(TASKS, key=lambda x: -r["tasks"][x]["gate_mean"]).index("pressure")) + 1
        abl_rows.append(dict(run=r["tag"], channels=r.get("channels", 7), encoder=r.get("encoder", "pooled"), seed=r["seed"],
                             spearman_gate_vs_player_drop=round(float(rho), 3), pressure_gate_rank=rank_p,
                             **{f"drop_{t}": round(ab[t]["drop_players"], 4) for t in TASKS}))
    if abl_rows:
        lines += ["## 门控均值对球员通道消融下降量（202 场，逐种子）", "", pd.DataFrame(abl_rows).to_markdown(index=False), ""]
        two = [x for x in abl_rows if x["run"] == "u1_2ch"]
        if two:
            rhos, ranks = [x["spearman_gate_vs_player_drop"] for x in two], [x["pressure_gate_rank"] for x in two]
            if len(two) < 3:
                verdict = f"只有 {len(two)} 个种子，还不能判"
            elif all(x >= 0.70 for x in rhos) and all(k <= 4 for k in ranks):
                verdict = "成立"
            elif all(x < 0.40 for x in rhos) or all(k >= 6 for k in ranks):
                verdict = "不成立"
            else:
                verdict = "不确定"
            lines += [f"两通道版按预先写下的标准（`memory_logs/2026-10-02_两通道统一模型_预先写下的判定标准.md`）判定  {verdict}。"
                      f"逐种子相关 {rhos}，Pressure 门控名次 {ranks}。", ""]

    # 基准运行按（编码器，通道数，标量是否加厚，门控代价，是否硬门控，种子）配对，跨机器对照 u1chk 不当基准
    cfg = lambda r: (r.get("encoder", "pooled"), r.get("channels", 7), bool(r.get("rich_scalars")), r.get("gate_penalty", 0), bool(r.get("hard_gate")), r["seed"])
    base = {}
    for r in runs:
        if r["fusion"] == "gate" and not r["shuffled"] and r["tag"] != "u1chk":
            assert cfg(r) not in base, f"两个运行的设置和种子完全相同，基准会被覆盖  {base[cfg(r)]['tag']} 和 {r['tag']}"
            base[cfg(r)] = r
    inter = [r for r in runs if r["shuffled"] and r["fusion"] == "gate" and cfg(r) in base]
    if inter:
        lines += ["## 干预  打乱某任务的空间输入后门控均值的变化（相对同编码器、同通道数、同种子的基准运行）", ""]
        rows = []
        for r in inter:
            b = base[cfg(r)]
            for t in TASKS:
                rank = lambda q: int(sorted(TASKS, key=lambda x: -q["tasks"][x]["gate_mean"]).index(t)) + 1
                rows.append(dict(run=r["tag"], seed=r["seed"], task=t, shuffled=t in r["shuffled"],
                                 gate_base=round(b["tasks"][t]["gate_mean"], 4), gate_shuffled_run=round(r["tasks"][t]["gate_mean"], 4),
                                 rank_base=rank(b), rank_shuffled_run=rank(r),
                                 gate_change=round(r["tasks"][t]["gate_mean"] - b["tasks"][t]["gate_mean"], 4),
                                 metric_base=round(metric(b, t), 4), metric_shuffled_run=round(metric(r, t), 4)))
        lines += [pd.DataFrame(rows).to_markdown(index=False), ""]

    # 门控对拼接只比同一套设置下的两个标签。2026-10-02 晚上查出，原来按“门控融合且没打乱”取门控运行，
    # 加了加厚标量、带代价、硬门控、两通道这些运行后，同一个种子会被后读入的运行覆盖，表里比的就不是 u1 了。
    for enc, gate_tag, concat_tag in [("pooled", "u1", "u1concat"), ("fcn", "u1fcn", "u1fcnconcat")]:
        g = {r["seed"]: r for r in runs if r["tag"] == gate_tag}
        c = {r["seed"]: r for r in runs if r["tag"] == concat_tag}
        seeds = sorted(set(g) & set(c))
        if seeds:
            lines += [f"## 同一统一模型里门控对拼接（编码器 {enc}，配对种子 {seeds}）", ""]
            rows = []
            for t in TASKS:
                d = np.array([metric(g[s], t) - metric(c[s], t) for s in seeds])
                rows.append(dict(task=t, metric="top1" if t == "dest" else "AUC", gate_minus_concat_mean=round(d.mean(), 4),
                                 min=round(d.min(), 4), max=round(d.max(), 4), seeds_gate_better=int((d > 0).sum())))
            lines += [pd.DataFrame(rows).to_markdown(index=False), ""]
    # 加厚标量的统一模型（u1rich）对同种子基准（u1），按 memory_logs/2026-10-02_加厚标量统一模型_预先写下的判定标准.md 判定
    rich = {r["seed"]: r for r in runs if r["tag"] == "u1rich"}
    b0 = {r["seed"]: r for r in runs if r["tag"] == "u1"}
    seeds = sorted(set(rich) & set(b0))
    if seeds:
        outc = [t for t in TASKS if t != "dest"]
        gm = lambda r, t: r["tasks"][t]["gate_mean"]
        rows, drops, ndown, dchg, drank = [], [], [], [], []
        for sd in seeds:
            ch = np.array([gm(rich[sd], t) - gm(b0[sd], t) for t in outc])
            drops.append(float(-ch.mean()))
            ndown.append(int((ch < 0).sum()))
            dchg.append(float(gm(rich[sd], "dest") - gm(b0[sd], "dest")))
            drank.append(int(sorted(TASKS, key=lambda x: -gm(rich[sd], x)).index("dest")) + 1)
            rows.append(dict(seed=sd, outcome_gate_avg_base=round(float(np.mean([gm(b0[sd], t) for t in outc])), 4),
                             outcome_gate_avg_rich=round(float(np.mean([gm(rich[sd], t) for t in outc])), 4), outcome_gate_avg_drop=round(drops[-1], 4),
                             tasks_down=f"{ndown[-1]}/7", dest_gate_base=round(gm(b0[sd], "dest"), 4), dest_gate_rich=round(gm(rich[sd], "dest"), 4),
                             dest_gate_change=round(dchg[-1], 4), dest_rank_rich=drank[-1]))
        lines += ["## 加厚标量的统一模型（u1rich）对同种子基准（u1）", "", pd.DataFrame(rows).to_markdown(index=False), ""]
        pj = CACHE_DIR / "pilot_third_view" / "rich_scalar_baseline_results.json"
        xgb = {t: v["S1_加手工汇总"]["mean"] for t, v in json.load(open(pj)).items()} if pj.exists() else {}
        rows, ok = [], 0
        for t in TASKS:
            mb, mr = np.mean([metric(b0[sd], t) for sd in seeds]), np.mean([metric(rich[sd], t) for sd in seeds])
            row = dict(task=t, metric="top1" if t == "dest" else "AUC", base=round(mb, 4), rich=round(mr, 4), rich_minus_base=round(mr - mb, 4),
                       gate_base=round(np.mean([gm(b0[sd], t) for sd in seeds]), 4), gate_rich=round(np.mean([gm(rich[sd], t) for sd in seeds]), 4),
                       reliance_base=round(float(np.mean([b0[sd]["tasks"][t].get("spatial_reliance", np.nan) for sd in seeds])), 4),
                       reliance_rich=round(np.mean([rich[sd]["tasks"][t]["spatial_reliance"] for sd in seeds]), 4))
            if t in xgb:
                row.update(rich_xgb=xgb[t], rich_minus_rich_xgb=round(mr - xgb[t], 4))
                ok += int(mr - xgb[t] >= -0.003)
            rows.append(row)
        lines += [pd.DataFrame(rows).to_markdown(index=False), "",
                  "reliance 两列是训练脚本在 229 场上算的整张量打乱下降量（skill 之差），只作参考，早期的基准运行没存这个量时显示 nan；其余列在 202 场上算。rich_xgb 是试验七加厚 XGBoost 五个种子的均值。", ""]
        if len(seeds) < 3:
            v_gate = f"只有 {len(seeds)} 个种子，还不能判"
        elif all(d >= 0.10 for d in drops) and all(k >= 5 for k in ndown) and all(abs(c) < 0.10 for c in dchg) and all(k == 1 for k in drank):
            v_gate = "成立"
        elif all(d < 0.05 for d in drops) or all(-c > d for c, d in zip(dchg, drops)):
            v_gate = "不成立"
        else:
            v_gate = "不确定"
        v_perf = (f"追平（{ok}/7 个任务的差不低于负 0.003）" if ok >= 6 else f"没追平（只有 {ok}/7 个任务的差不低于负 0.003）") if xgb else "缺加厚 XGBoost 的结果文件"
        lines += [f"按预先写下的标准判定。门控这一问  {v_gate}（七个结果类任务门控平均的下降 {[round(d, 3) for d in drops]}，下降的任务数 {ndown}，"
                  f"落点门控变化 {[round(c, 3) for c in dchg]}，落点名次 {drank}）。成绩这一问  {v_perf}。", ""]

    # 拼接版落点干预对照，按 memory_logs/2026-10-02_拼接版落点干预对照_预先写下的判定标准.md
    tag_seed = {(r["tag"], r["seed"]): r for r in runs}
    cs = sorted(sd for (tg, sd) in tag_seed if tg == "u1concat_shuf_dest" and all((x, sd) in tag_seed for x in ("u1", "shuf_dest", "u1concat")))
    if cs:
        outc = [t for t in TASKS if t != "dest"]
        rows, diffs = [], []
        for sd in cs:
            dg = {t: metric(tag_seed[("shuf_dest", sd)], t) - metric(tag_seed[("u1", sd)], t) for t in TASKS}
            dc = {t: metric(tag_seed[("u1concat_shuf_dest", sd)], t) - metric(tag_seed[("u1concat", sd)], t) for t in TASKS}
            ag, ac = float(np.mean([dg[t] for t in outc])), float(np.mean([dc[t] for t in outc]))
            diffs.append(ag - ac)  # 正数表示拼接版掉得更多
            for name, d, avg in [("门控", dg, ag), ("拼接", dc, ac)]:
                rows.append(dict(seed=sd, fusion=name, other7_avg_change=round(avg, 4), **{t: round(d[t], 4) for t in TASKS}))
        lines += ["## 落点干预下其余任务被拖累多少，门控对拼接", "", pd.DataFrame(rows).to_markdown(index=False), "",
                  "各列是干预运行减同种子同融合方式的基准运行，落点是 top-1 之差，其余是 AUC 之差。", ""]
        if len(cs) < 3:
            v = f"只有 {len(cs)} 个种子，还不能判"
        elif all(d >= 0.005 for d in diffs):
            v = "成立"
        elif all(abs(d) <= 0.002 for d in diffs):
            v = "不成立"
        else:
            v = "不确定"
        lines += [f"按预先写下的标准判定  {v}。拼接版比门控版多掉的量（逐种子） {[round(d, 4) for d in diffs]}。", ""]

    # 门控代价实验（软门控）和硬门控实验，分别按 memory_logs/ 下两份预先写下的判定标准
    pen = [r for r in runs if r.get("gate_penalty", 0) > 0 and r["fusion"] == "gate" and not r["shuffled"] and r.get("encoder", "pooled") == "pooled"]
    if pen:
        outc = [t for t in TASKS if t != "dest"]
        gmean = lambda r, t: r["tasks"][t]["gate_mean"]
        gains = single_task_gains()
        need = loss_unit_need()
        lines += ["## 门控代价实验（损失里加 λ × 门控均值），软门控和硬门控", "",
                  "spearman_gate_vs_loss_need 是门控对“单任务 M4 相对 M2 每事件对数损失下降量”的相关，"
                  "各任务的下降量是 " + "，".join(f"{t} {need[t]:.4f}" for t in TASKS) + "。tiers 两列是分层检验（高层落点和 Pressure，中层 Interception 和 Tackle，低层其余四个）。"
                  "gate_change_last3 是被选中那一轮相对之前三轮的结果类任务门控平均变化，绝对值超过 0.03 记为没走平。"
                  "gate_zero 两列是测试时把门强制关上后的下降量（202 场，结果类任务取平均，落点单列），量的是没有空间分支行不行；硬门控的门控均值是开门维度的比例，all_closed 两列是 64 个维度全部关着的事件占比，open_g_below_06 是开着的维度里开门概率低于 0.6 的比例。", ""]

        def abl(r):
            f = U1 / f"{r['tag']}_s{r['seed']}_ablation.json"
            return json.load(open(f)) if f.exists() else None

        def extra_cols(r):
            ab = abl(r)
            gd = {t: gmean(r, t) for t in TASKS}
            t_ok, t_viol = tier_check(gd)
            pl = plateau(r)
            c = dict(spearman_gate_vs_loss_need=round(float(spearmanr([gd[t] for t in TASKS], [need[t] for t in TASKS])[0]), 2),
                     tiers_ordered="是" if t_ok else "否", tier_violations=f"{t_viol}/20", gate_change_last3="无记录" if pl is None else round(pl, 3))
            if ab and "drop_gate_zero" in ab[TASKS[0]]:
                gz = {t: ab[t]["drop_gate_zero"] for t in TASKS}
                # 门控对自身强制关门下降量的相关，下降量用每事件损失的上升量（和需要量同单位）；旧的消融文件没有这个量时退回指标下降量
                gzl = {t: ab[t].get("loss_up_gate_zero", ab[t]["drop_gate_zero"]) for t in TASKS}
                c.update(gate_zero_outcome_avg=round(float(np.mean([gz[t] for t in outc])), 4), gate_zero_dest=round(gz["dest"], 4),
                         spearman_gate_vs_gate_zero=round(float(spearmanr([gd[t] for t in TASKS], [gzl[t] for t in TASKS])[0]), 2),
                         drop_all_outcome_avg=round(float(np.mean([ab[t]["drop_all"] for t in outc])), 4),
                         spearman_gate_vs_drop_players=round(float(spearmanr([gd[t] for t in TASKS], [ab[t]["drop_players"] for t in TASKS])[0]), 2))
                if "all_closed_share" in ab[TASKS[0]]:
                    c.update(all_closed_outcome_avg=round(float(np.mean([ab[t]["all_closed_share"] for t in outc])), 3), all_closed_dest=round(ab["dest"]["all_closed_share"], 3),
                             open_g_below_06=round(float(np.nanmean([ab[t]["open_g_below_06_share"] if ab[t]["open_g_below_06_share"] is not None else np.nan for t in TASKS])), 2))
            else:
                c.update(gate_zero_outcome_avg="未做消融")
            return c, ab

        for hard_flag, rich_flag in [(False, True), (False, False), (True, True), (True, False)]:
            name = ("硬门控" if hard_flag else "软门控") + "，" + ("标量加厚" if rich_flag else "标量薄")
            base_tag = "u1rich" if rich_flag else "u1"
            base_by_seed = {r["seed"]: r for r in runs if r["tag"] == base_tag}
            grp = [r for r in pen if bool(r.get("rich_scalars")) == rich_flag and bool(r.get("hard_gate")) == hard_flag]
            if not grp:
                continue
            path = sorted([r for r in grp if r["seed"] == 0], key=lambda r: r["gate_penalty"])
            rows, chosen = [], None
            if 0 in base_by_seed:
                path = [base_by_seed[0]] + path
            for r in path:
                lam_ = r.get("gate_penalty", 0)
                okv = 0 in base_by_seed and r["best_val_score"] >= base_by_seed[0]["best_val_score"] - 0.002
                if lam_ > 0 and okv:
                    chosen = lam_
                rho8 = spearmanr([gmean(r, t) for t in TASKS], [gains[t][0] for t in TASKS])[0]
                c, _ = extra_cols(r)
                rows.append(dict(lam=lam_, epochs=r["n_epochs"], val_score=round(r["best_val_score"], 4), within_0002="是" if okv else "否",
                                 outcome_gate_avg=round(float(np.mean([gmean(r, t) for t in outc])), 3), spearman_gate_vs_gain=round(float(rho8), 2), **c,
                                 **{f"g_{t}": round(gmean(r, t), 3) for t in TASKS}, **{f"m_{t}": round(metric(r, t), 4) for t in TASKS}))
            lines += [f"### {name}，探索步（种子 0 的路径，只用来选 λ；λ 为 0 的一行是无代价的软门控基准 {base_tag}）", "", pd.DataFrame(rows).to_markdown(index=False), "",
                      f"g_ 开头是门控均值，m_ 开头是测试指标（落点是 top-1，其余 AUC）。按规则选出的 λ 是 {chosen}（验证得分比基准低不超过 0.002 的最大 λ）。", ""]
            if chosen is None:
                continue
            conf = {r["seed"]: r for r in grp if r["gate_penalty"] == chosen and r["seed"] in (1, 2, 3)}
            rows, flags = [], []
            for sd in sorted(conf):
                r, b = conf[sd], base_by_seed.get(sd)
                og, dg = float(np.mean([gmean(r, t) for t in outc])), gmean(r, "dest")
                drank = int(sorted(TASKS, key=lambda x: -gmean(r, x)).index("dest")) + 1
                rho8 = float(spearmanr([gmean(r, t) for t in TASKS], [gains[t][0] for t in TASKS])[0])
                rho7 = float(spearmanr([gmean(r, t) for t in outc], [gains[t][0] for t in outc])[0])
                n_ok = sum(metric(r, t) - metric(b, t) >= (-0.005 if t == "dest" else -0.003) for t in TASKS) if b is not None else None
                c, ab = extra_cols(r)
                rows.append(dict(seed=sd, tag=r["tag"], epochs=r["n_epochs"], outcome_gate_avg=round(og, 3), dest_gate=round(dg, 3), dest_rank=drank, spearman_8=round(rho8, 2), spearman_7=round(rho7, 2),
                                 tasks_within_tolerance=f"{n_ok}/8" if n_ok is not None else "缺基准", **c))
                flags.append(dict(og=og, dg=dg, drank=drank, rho8=rho8, rho7=rho7, n_ok=n_ok, c=c, has_ab=ab is not None and "drop_gate_zero" in ab[TASKS[0]]))
            if rows:
                lines += [f"### {name}，确认步（λ 等于 {chosen}，种子 1、2、3）", "", pd.DataFrame(rows).to_markdown(index=False), ""]
            if not (len(flags) == 3 and all(f["n_ok"] is not None for f in flags)):
                lines += [f"确认步还缺种子或缺基准（现有 {len(flags)} 个），暂不判。", ""]
                continue
            if (rich_flag or hard_flag) and not all(f["has_ab"] for f in flags):
                lines += ["确认步缺 202 场消融，暂不判。", ""]
                continue
            if not hard_flag and rich_flag:
                if all(f["og"] <= 0.10 and f["dg"] >= 0.50 and f["drank"] == 1 and f["n_ok"] >= 7 and f["c"]["drop_all_outcome_avg"] <= 0.01 for f in flags):
                    v = "成立"
                elif all(f["og"] > 0.20 for f in flags) or all(f["n_ok"] <= 5 for f in flags) or all(f["dg"] < f["og"] for f in flags):
                    v = "不成立"
                else:
                    v = "不确定"
                lines += [f"甲（软门控，标量加厚加代价）按预先写下的标准判定  {v}。", ""]
            elif not hard_flag:
                if all(f["rho8"] >= 0.70 and f["rho7"] >= 0.60 and f["n_ok"] >= 7 for f in flags):
                    v = "成立"
                elif all(f["rho8"] < 0.40 for f in flags):
                    v = "不成立"
                else:
                    v = "不确定"
                lines += [f"乙（软门控，标量薄加代价）按预先写下的标准判定  {v}。", ""]
            elif rich_flag:
                if all(f["og"] <= 0.10 and f["dg"] >= 0.25 and f["drank"] == 1 and f["dg"] >= 3 * f["og"] and f["n_ok"] >= 7
                       and f["c"]["gate_zero_outcome_avg"] <= 0.005 and f["c"]["gate_zero_dest"] >= 0.02 for f in flags):
                    v = "成立"
                elif all(f["og"] > 0.20 for f in flags) or all(f["n_ok"] <= 5 for f in flags) or all(f["c"]["gate_zero_outcome_avg"] > 0.01 for f in flags):
                    v = "不成立"
                else:
                    v = "不确定"
                lines += [f"甲二（硬门控，标量加厚加代价）按预先写下的标准判定  {v}。", ""]
            else:
                if all(f["c"]["spearman_gate_vs_loss_need"] >= 0.60 and f["c"]["spearman_gate_vs_gate_zero"] >= 0.80 and f["n_ok"] >= 7 for f in flags):
                    v = "成立"
                elif all(f["c"]["spearman_gate_vs_loss_need"] < 0.30 for f in flags) or all(f["c"]["spearman_gate_vs_gate_zero"] < 0.50 for f in flags):
                    v = "不成立"
                else:
                    v = "不确定"
                lines += [f"乙二（硬门控，标量薄加代价）按预先写下的标准判定  {v}。", ""]

    lines += single_task_table()
    lines += single_task_table("single_clean_rich", "同数据单任务模型，标量加厚版（标量特征加 15 个从张量手工汇总的量，202 场，跨种子均值 ± 标准差）")
    OUT.write_text("\n".join(lines), encoding="utf8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
