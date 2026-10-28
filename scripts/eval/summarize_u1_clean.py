#!/usr/bin/env python3
"""
summarize_u1_clean.py, results of the unified model on the clean test matches
=============================================================================
Reads the outputs of train_unified_clean.py and summarises them per run tag. Every test metric and gate mean is
recomputed from the per-event predictions on the 202 clean test matches of 2024/25.
  Per tag            test metric and gate mean of every task, mean, SD, min and max across seeds, and the rank of
                     the gate mean in each seed.
  Gate and gain      Spearman correlation between the task gate means of the unified model and the single-task CNN
                     gain (M4 minus M2), per seed and on the seed means. For dest the gain is a difference in
                     normalised log loss.
  Interventions      runs with --shuffle_spatial against the baseline run with the same settings and seed, change of
                     the gate mean of the shuffled task and of the other tasks.
  Fusion control     runs with --fusion concat against the gated run with the same seed, per-task metric differences.
  Single-task models four single-task models trained on the same data, scores on the 202 matches and paired
                     differences with the same seed.
  Gate penalty       runs with --gate_penalty. The seed 0 path selects lambda, seeds 1 to 3 are judged by the rule
                     fixed in advance.
  Ablation control   runs with an ablation.json file. Spearman correlation between the gate mean and the loss from
                     shuffling the player channels; the two-channel model is judged by the rule fixed in advance.
The rules are listed in results/decision_rules.md.

Inputs   data/cache/u1_clean/*.json and *.npz
         data/cache/single_clean/*.jsonl and preds/ (single-task models on the same data, used when present)
         data/cache/single_clean_rich/ (the same with enriched scalars)
         data/cache/multiseed/ and data/cache/multiseed_fixed/ jsonl and preds (used only where single_clean is missing)
         data/raw_scan/match_state_2425.parquet
Outputs  results/u1_clean_summary.md

Usage
  PYTHONPATH=. python -u scripts/eval/summarize_u1_clean.py

Last modified 2026-10-28
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
OUT = Path("results/u1_clean_summary.md")
TASKS = ["pass", "dest", "shot", "interception", "ball_recovery", "pressure", "dribble", "tackle"]


def clean_matches():
    """Clean test matches, the 202 matches of 2024/25 in the normal 360 orientation state (at most 20% of frames
    reversed) with at least 70% of shots carrying a frame. The other 27 matches in the normal state carry frames for
    shots and out-of-play passes only for some outcomes, so they cannot serve as test matches."""
    st = pd.read_parquet("data/raw_scan/match_state_2425.parquet")
    return set(st.loc[(st["state"] == "normal") & (st["shot_frame_pct"] >= 70), "match_id"].astype(int))


def recompute(r, clean):
    """Recompute the test metrics and gate means of every task from the per-event predictions on the clean matches,
    replacing the values in the json, which cover 229 matches"""
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
    """CNN gain of the single-task models, M4 minus M2, paired by seed on the clean matches and averaged.
    Returns {task: (gain, number of seeds, note)}"""
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
                note = "pressure label from the possession_team definition, not the 5 s label" if t == "pressure" else ""
                out[t] = (float(np.mean(g)), len(g), note)
    # Single-task results on the same data as the unified model (train_single_clean.py) take precedence
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
            note = "difference in normalised log loss, not comparable in size with AUC differences, used only for ranking" if t == "dest" else ""
            out[t] = (float(np.mean([m4[s] - m2[s] for s in seeds])), len(seeds), ("same-data single-task model. " + note) if note else "same-data single-task model")
    return out


def loss_unit_need():
    """Need on the same scale as the gate penalty, the per-event log-loss reduction of the single-task M4 over M2
    (202 matches, mean over five seeds), times 1/ln96 for dest. The per-event training predictions were not stored,
    so the test set stands in for them. Returns {task: reduction}"""
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
    """Tier test. g is {task: gate mean}. Returns (whether the three tier means decrease in order, number of the
    20 cross-tier task pairs whose order is reversed)"""
    means = [float(np.mean([g[t] for t in tier])) for tier in TIERS]
    viol = sum(g[a] <= g[b] for i in range(3) for j in range(i + 1, 3) for a in TIERS[i] for b in TIERS[j])
    return means[0] > means[1] > means[2], int(viol)


def plateau(r):
    """Change of the mean validation gate of the seven outcome tasks at the selected epoch against three epochs
    earlier. Returns None when the run has no per-epoch record"""
    h = r.get("history", [])
    if not h or h[0].get("val_gate") is None:
        return None
    best = int(np.argmax([x["val_score"] for x in h]))
    if best < 3:
        return float("nan")
    avg = lambda i: float(np.mean([h[i]["val_gate"][t] for t in TASKS if t != "dest"]))
    return avg(best) - avg(best - 3)


def single_task_means(dirname, model):
    """Mean AUC over seeds of one single-task model on the 202 matches, per outcome task, rounded to four decimals
    (the same per-seed AUC as single_task_table)"""
    clean = clean_matches()
    sc = CACHE_DIR / dirname
    out = {}
    for t in TASKS:
        if t == "dest" or not (sc / f"{t}.jsonl").exists():
            continue
        meta = np.load(sc / "preds" / f"{t}_test_meta.npz", allow_pickle=True)
        okc = np.array([int(m) in clean for m in meta["match_id"]])
        yy = meta["y"].astype(int)
        v = [float(roc_auc_score(yy[okc], np.load(f)["prob"][okc])) for f in sorted(glob.glob(str(sc / "preds" / f"{t}_{model}_s*.npz")))]
        if v:
            out[t] = round(float(np.mean(v)), 4)
    return out


def single_task_table(dirname="single_clean", title="Single-task models on the same data (202 matches, mean ± SD across seeds)"):
    """Per-model scores and paired differences of the single-task models on the 202 matches. dirname is single_clean
    (thin scalars) or single_clean_rich (enriched scalars). Returns a list of markdown lines"""
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
        row = dict(task=t, metric="skill (top1)" if t == "dest" else "AUC", n_test=int(okc.sum()))
        for m in models:
            v = np.array([val[k][0] for k in sorted(val) if k[0] == m])
            if len(v):
                cell = f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}"
                if t == "dest":
                    cell += f" ({np.mean([val[k][1] for k in val if k[0] == m]):.4f})"
                row[m] = cell + f" n={len(v)}"
        rows.append(row)
        for a, b in [("M4_CNN_Full", "M2_MLP_360"), ("G1_Gating", "M2_MLP_360"), ("G1_Gating", "M4_CNN_Full"), ("M4_CNN_Full", "B2_XGB_360")]:
            seeds = sorted(set(k[1] for k in val if k[0] == a) & set(k[1] for k in val if k[0] == b))
            if not seeds:
                continue
            d = np.array([val[(a, s_)][0] - val[(b, s_)][0] for s_ in seeds])
            diffs.append(dict(task=t, pair=f"{a} minus {b}", mean=round(d.mean(), 4), sd=round(d.std(ddof=1), 4) if len(d) > 1 else np.nan,
                              min=round(d.min(), 4), max=round(d.max(), 4), seeds_positive=f"{int((d > 0).sum())}/{len(d)}"))
    if not rows:
        return []
    return [f"## {title}", "",
            "For dest the value is skill (1 minus log loss over ln96), with top-1 in parentheses. XGBoost is not fitted to the destination task.", "",
            pd.DataFrame(rows).to_markdown(index=False), "", "### Paired differences, same seed", "", pd.DataFrame(diffs).to_markdown(index=False), ""]


def main():
    runs = load_runs()
    lines = ["# Unified model results on the 202 test matches of 2024/25", "", "Generated by `scripts/eval/summarize_u1_clean.py`. Test metrics and gate means are recomputed from the per-event predictions on the 202 test matches of 2024/25 whose freeze frames passed the consistency check."
             " (At training time the test season held the 229 matches of 2024/25 whose freeze-frame teammate flags follow the provider specification; the 27 of them whose shots and out-of-play passes carry freeze frames only for some outcomes are left out when scoring.)", ""]
    if not runs:
        OUT.write_text("\n".join(lines + ["No results yet."]), encoding="utf8")
        print("no runs")
        return
    gains = single_task_gains()
    if gains and all(g[2].startswith("same-data single-task model") for g in gains.values()):
        lines += ["In the run tables, single_task_cnn_gain is the gain of the single-task CNN (M4_CNN_Full) over the scalar MLP (M2_MLP_360) trained on the same data, "
                  "paired by seed and averaged over gain_seeds seeds. It is a difference in AUC, and for dest a difference in skill that is used only for ranking.", ""]
    elif gains:
        lines += ["Source of the single_task_cnn_gain column. " + "; ".join(f"{t} {g[2]}" for t, g in gains.items() if g[2]), ""]
    tags = sorted(set(r["tag"] for r in runs))
    by = {tag: sorted([r for r in runs if r["tag"] == tag], key=lambda r: r["seed"]) for tag in tags}

    for tag in tags:
        rs = by[tag]
        r0 = rs[0]
        lines += [f"## Run {tag} (fusion {r0['fusion']}, encoder {r0.get('encoder', 'pooled')}, shuffled tasks {','.join(r0['shuffled']) or 'none'}, seeds {[r['seed'] for r in rs]})", ""]
        rows = []
        for t in TASKS:
            m = np.array([metric(r, t) for r in rs])
            row = dict(task=t, metric="top1" if t == "dest" else "AUC", test_mean=round(m.mean(), 4),
                       test_sd=round(m.std(ddof=1), 4) if len(m) > 1 else np.nan)
            if t == "dest":
                # dest also gets skill (1 minus log loss over ln96) and log loss, on the same scale as the rule baselines and the single-task table
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
            for name, ts in [("with destination", common), ("AUC tasks only", auc_only)]:
                if len(ts) < 4:
                    continue
                per = []
                for r in rs:
                    rho, _ = spearmanr([r["tasks"][t]["gate_mean"] for t in ts], [gains[t][0] for t in ts])
                    per.append(rho)
                gm = [np.mean([r["tasks"][t]["gate_mean"] for r in rs]) for t in ts]
                rho, p = spearmanr(gm, [gains[t][0] for t in ts])
                lines += [f"Spearman correlation of the gate mean with the single-task CNN gain ({name}, {len(ts)} tasks). Per seed {', '.join(f'{x:.2f}' for x in per)}; on the seed means {rho:.2f} (p {p:.3f})", ""]

    # Gate mean against the loss from shuffling the player channels (ablate_u1_saved.py, 202 matches), Spearman and Pressure gate rank per seed
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
        lines += ["## Gate mean against the loss from shuffling the player channels (202 matches, per seed)", "", pd.DataFrame(abl_rows).to_markdown(index=False), ""]
        two = [x for x in abl_rows if x["run"] == "u1_2ch"]
        if two:
            rhos, ranks = [x["spearman_gate_vs_player_drop"] for x in two], [x["pressure_gate_rank"] for x in two]
            if len(two) < 3:
                verdict = f"only {len(two)} seeds, not judged"
            elif all(x >= 0.70 for x in rhos) and all(k <= 4 for k in ranks):
                verdict = "established"
            elif all(x < 0.40 for x in rhos) or all(k >= 6 for k in ranks):
                verdict = "not supported"
            else:
                verdict = "inconclusive"
            lines += [f"Two-channel model, verdict under the rule fixed in advance (`results/decision_rules.md`), {verdict}."
                      f" Correlation per seed {rhos}, Pressure gate rank {ranks}.", ""]

    # Baseline runs are matched on (encoder, channels, enriched scalars, gate penalty, hard gate, seed); the reproducibility check run u1chk is not a baseline
    cfg = lambda r: (r.get("encoder", "pooled"), r.get("channels", 7), bool(r.get("rich_scalars")), r.get("gate_penalty", 0), bool(r.get("hard_gate")), r["seed"])
    base = {}
    for r in runs:
        if r["fusion"] == "gate" and not r["shuffled"] and r["tag"] != "u1chk":
            assert cfg(r) not in base, f"two runs share settings and seed, the baseline would be overwritten  {base[cfg(r)]['tag']} and {r['tag']}"
            base[cfg(r)] = r
    inter = [r for r in runs if r["shuffled"] and r["fusion"] == "gate" and cfg(r) in base]
    if inter:
        lines += ["## Interventions, change of the gate mean after shuffling the spatial input of one task (against the baseline run with the same encoder, number of channels and seed)", ""]
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

    # Gate against concatenation compares only the two tags with otherwise identical settings. Selecting the gated run as
    # "gate fusion, nothing shuffled" would let runs with enriched scalars, a penalty, a hard gate or two channels overwrite u1 for the same seed
    for enc, gate_tag, concat_tag in [("pooled", "u1", "u1concat"), ("fcn", "u1fcn", "u1fcnconcat")]:
        g = {r["seed"]: r for r in runs if r["tag"] == gate_tag}
        c = {r["seed"]: r for r in runs if r["tag"] == concat_tag}
        seeds = sorted(set(g) & set(c))
        if seeds:
            lines += [f"## Gate against concatenation in the same unified model (encoder {enc}, paired seeds {seeds})", ""]
            rows = []
            for t in TASKS:
                d = np.array([metric(g[s], t) - metric(c[s], t) for s in seeds])
                rows.append(dict(task=t, metric="top1" if t == "dest" else "AUC", gate_minus_concat_mean=round(d.mean(), 4),
                                 min=round(d.min(), 4), max=round(d.max(), 4), seeds_gate_better=int((d > 0).sum())))
            lines += [pd.DataFrame(rows).to_markdown(index=False), ""]
    # Unified model with enriched scalars (u1rich) against the baseline with the same seed (u1), judged by the rule fixed in advance
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
        lines += ["## Unified model with enriched scalars (u1rich) against the baseline with the same seed (u1)", "", pd.DataFrame(rows).to_markdown(index=False), ""]
        xgb = single_task_means("single_clean_rich", "B2_XGB_360")
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
                  "The two reliance columns are the drop under a full tensor shuffle (difference in skill) computed by the training script on 229 matches, for reference only; they show nan for baseline runs that did not store this quantity. The other columns are computed on 202 matches. rich_xgb is the mean of five seeds of enriched XGBoost.", ""]
        if len(seeds) < 3:
            v_gate = f"only {len(seeds)} seeds, not judged"
        elif all(d >= 0.10 for d in drops) and all(k >= 5 for k in ndown) and all(abs(c) < 0.10 for c in dchg) and all(k == 1 for k in drank):
            v_gate = "established"
        elif all(d < 0.05 for d in drops) or all(-c > d for c, d in zip(dchg, drops)):
            v_gate = "not supported"
        else:
            v_gate = "inconclusive"
        v_perf = (f"matched ({ok}/7 tasks with a difference of at least -0.003)" if ok >= 6 else f"not matched (only {ok}/7 tasks with a difference of at least -0.003)") if xgb else "enriched XGBoost result file missing"
        lines += [f"Verdicts under the rules fixed in advance. Gate question, {v_gate} (drop of the mean gate of the seven outcome tasks {[round(d, 3) for d in drops]}, outcome tasks whose gate fell {ndown}, "
                  f"destination gate change {[round(c, 3) for c in dchg]}, destination rank {drank}). Accuracy question, {v_perf}.", ""]

    # Concatenation control under the destination intervention, judged by the rule fixed in advance
    tag_seed = {(r["tag"], r["seed"]): r for r in runs}
    cs = sorted(sd for (tg, sd) in tag_seed if tg == "u1concat_shuf_dest" and all((x, sd) in tag_seed for x in ("u1", "shuf_dest", "u1concat")))
    if cs:
        outc = [t for t in TASKS if t != "dest"]
        rows, diffs = [], []
        for sd in cs:
            dg = {t: metric(tag_seed[("shuf_dest", sd)], t) - metric(tag_seed[("u1", sd)], t) for t in TASKS}
            dc = {t: metric(tag_seed[("u1concat_shuf_dest", sd)], t) - metric(tag_seed[("u1concat", sd)], t) for t in TASKS}
            ag, ac = float(np.mean([dg[t] for t in outc])), float(np.mean([dc[t] for t in outc]))
            diffs.append(ag - ac)  # positive means the concatenation model loses more
            for name, d, avg in [("gate", dg, ag), ("concat", dc, ac)]:
                rows.append(dict(seed=sd, fusion=name, other7_avg_change=round(avg, 4), **{t: round(d[t], 4) for t in TASKS}))
        lines += ["## Loss of the other tasks under the destination intervention, gate against concatenation", "", pd.DataFrame(rows).to_markdown(index=False), "",
                  "Each column is the intervention run minus the baseline run with the same seed and fusion, as a difference in top-1 for dest and in AUC otherwise.", ""]
        if len(cs) < 3:
            v = f"only {len(cs)} seeds, not judged"
        elif all(d >= 0.005 for d in diffs):
            v = "established"
        elif all(abs(d) <= 0.002 for d in diffs):
            v = "not supported"
        else:
            v = "inconclusive"
        lines += [f"Verdict under the rule fixed in advance, {v}. Extra loss of the concatenation model over the gated model (per seed) {[round(d, 4) for d in diffs]}.", ""]

    # Gate penalty experiments (soft gate) and hard gate experiments, judged by the rules fixed in advance
    pen = [r for r in runs if r.get("gate_penalty", 0) > 0 and r["fusion"] == "gate" and not r["shuffled"] and r.get("encoder", "pooled") == "pooled"]
    if pen:
        outc = [t for t in TASKS if t != "dest"]
        gmean = lambda r, t: r["tasks"][t]["gate_mean"]
        gains = single_task_gains()
        need = loss_unit_need()
        lines += ["## Gate penalty experiments (λ × gate mean added to the loss), soft and hard gate", "",
                  "spearman_gate_vs_loss_need is the correlation of the gate with the per-event log-loss reduction of the single-task M4 over M2; "
                  "the reductions per task are " + ", ".join(f"{t} {need[t]:.4f}" for t in TASKS) + ". The two tiers columns are a tier test (upper tier destination and Pressure, middle tier Interception and Tackle, lower tier the other four). "
                  "gate_change_last3 is the change of the mean outcome-task gate at the selected epoch against three epochs earlier; an absolute value above 0.03 counts as not settled. "
                  "The two gate_zero columns are the drop when the gate is forced closed at test time (202 matches, mean over outcome tasks, destination separately), which measures whether the model works without the spatial branch. For the hard gate the gate mean is the share of open dimensions, the two all_closed columns are the share of events with all 64 dimensions closed, and open_g_below_06 is the share of open dimensions whose opening probability is below 0.6.", ""]

        def abl(r):
            f = U1 / f"{r['tag']}_s{r['seed']}_ablation.json"
            return json.load(open(f)) if f.exists() else None

        def extra_cols(r):
            ab = abl(r)
            gd = {t: gmean(r, t) for t in TASKS}
            t_ok, t_viol = tier_check(gd)
            pl = plateau(r)
            c = dict(spearman_gate_vs_loss_need=round(float(spearmanr([gd[t] for t in TASKS], [need[t] for t in TASKS])[0]), 2),
                     tiers_ordered="yes" if t_ok else "no", tier_violations=f"{t_viol}/20", gate_change_last3="not stored" if pl is None else round(pl, 3))
            if ab and "drop_gate_zero" in ab[TASKS[0]]:
                gz = {t: ab[t]["drop_gate_zero"] for t in TASKS}
                # correlation of the gate with its own forced-closed loss, measured as the per-event loss increase (same unit as the need); ablation files without it fall back to the metric drop
                gzl = {t: ab[t].get("loss_up_gate_zero", ab[t]["drop_gate_zero"]) for t in TASKS}
                c.update(gate_zero_outcome_avg=round(float(np.mean([gz[t] for t in outc])), 4), gate_zero_dest=round(gz["dest"], 4),
                         spearman_gate_vs_gate_zero=round(float(spearmanr([gd[t] for t in TASKS], [gzl[t] for t in TASKS])[0]), 2),
                         drop_all_outcome_avg=round(float(np.mean([ab[t]["drop_all"] for t in outc])), 4),
                         spearman_gate_vs_drop_players=round(float(spearmanr([gd[t] for t in TASKS], [ab[t]["drop_players"] for t in TASKS])[0]), 2))
                if "all_closed_share" in ab[TASKS[0]]:
                    c.update(all_closed_outcome_avg=round(float(np.mean([ab[t]["all_closed_share"] for t in outc])), 3), all_closed_dest=round(ab["dest"]["all_closed_share"], 3),
                             open_g_below_06=round(float(np.nanmean([ab[t]["open_g_below_06_share"] if ab[t]["open_g_below_06_share"] is not None else np.nan for t in TASKS])), 2))
            else:
                c.update(gate_zero_outcome_avg="not run")
            return c, ab

        for hard_flag, rich_flag in [(False, True), (False, False), (True, True), (True, False)]:
            name = ("Hard gate" if hard_flag else "Soft gate") + ", " + ("enriched scalars" if rich_flag else "thin scalars")
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
                rows.append(dict(lam=lam_, epochs=r["n_epochs"], val_score=round(r["best_val_score"], 4), within_0002="yes" if okv else "no",
                                 outcome_gate_avg=round(float(np.mean([gmean(r, t) for t in outc])), 3), spearman_gate_vs_gain=round(float(rho8), 2), **c,
                                 **{f"g_{t}": round(gmean(r, t), 3) for t in TASKS}, **{f"m_{t}": round(metric(r, t), 4) for t in TASKS}))
            lines += [f"### {name}, selection step (path of seed 0, used only to choose λ; the row with λ 0 is the soft-gate baseline without penalty, {base_tag})", "", pd.DataFrame(rows).to_markdown(index=False), "",
                      f"Columns g_ are gate means and m_ are test metrics (top-1 for dest, AUC otherwise). {'The λ chosen by the rule is ' + str(chosen) if chosen is not None else 'No λ meets the rule'} (the largest λ whose validation score is at most 0.002 below the baseline).", ""]
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
                                 tasks_within_tolerance=f"{n_ok}/8" if n_ok is not None else "no baseline", **c))
                flags.append(dict(og=og, dg=dg, drank=drank, rho8=rho8, rho7=rho7, n_ok=n_ok, c=c, has_ab=ab is not None and "drop_gate_zero" in ab[TASKS[0]]))
            if rows:
                lines += [f"### {name}, confirmation step (λ = {chosen}, seeds 1, 2, 3)", "", pd.DataFrame(rows).to_markdown(index=False), ""]
            if not (len(flags) == 3 and all(f["n_ok"] is not None for f in flags)):
                lines += [f"Confirmation step incomplete, seeds or baseline missing ({len(flags)} available), not judged.", ""]
                continue
            if (rich_flag or hard_flag) and not all(f["has_ab"] for f in flags):
                lines += ["Confirmation step lacks the ablation on 202 matches, not judged.", ""]
                continue
            if not hard_flag and rich_flag:
                if all(f["og"] <= 0.10 and f["dg"] >= 0.50 and f["drank"] == 1 and f["n_ok"] >= 7 and f["c"]["drop_all_outcome_avg"] <= 0.01 for f in flags):
                    v = "established"
                elif all(f["og"] > 0.20 for f in flags) or all(f["n_ok"] <= 5 for f in flags) or all(f["dg"] < f["og"] for f in flags):
                    v = "not supported"
                else:
                    v = "inconclusive"
                lines += [f"Soft gate with penalty, enriched scalars, verdict under the rule fixed in advance, {v}.", ""]
            elif not hard_flag:
                if all(f["rho8"] >= 0.70 and f["rho7"] >= 0.60 and f["n_ok"] >= 7 for f in flags):
                    v = "established"
                elif all(f["rho8"] < 0.40 for f in flags):
                    v = "not supported"
                else:
                    v = "inconclusive"
                lines += [f"Soft gate with penalty, thin scalars, verdict under the rule fixed in advance, {v}.", ""]
            elif rich_flag:
                if all(f["og"] <= 0.10 and f["dg"] >= 0.25 and f["drank"] == 1 and f["dg"] >= 3 * f["og"] and f["n_ok"] >= 7
                       and f["c"]["gate_zero_outcome_avg"] <= 0.005 and f["c"]["gate_zero_dest"] >= 0.02 for f in flags):
                    v = "established"
                elif all(f["og"] > 0.20 for f in flags) or all(f["n_ok"] <= 5 for f in flags) or all(f["c"]["gate_zero_outcome_avg"] > 0.01 for f in flags):
                    v = "not supported"
                else:
                    v = "inconclusive"
                lines += [f"Hard gate with penalty, enriched scalars, verdict under the rule fixed in advance, {v}.", ""]
            else:
                if all(f["c"]["spearman_gate_vs_loss_need"] >= 0.60 and f["c"]["spearman_gate_vs_gate_zero"] >= 0.80 and f["n_ok"] >= 7 for f in flags):
                    v = "established"
                elif all(f["c"]["spearman_gate_vs_loss_need"] < 0.30 for f in flags) or all(f["c"]["spearman_gate_vs_gate_zero"] < 0.50 for f in flags):
                    v = "not supported"
                else:
                    v = "inconclusive"
                lines += [f"Hard gate with penalty, thin scalars, verdict under the rule fixed in advance, {v}.", ""]

    lines += single_task_table()
    lines += single_task_table("single_clean_rich", "Single-task models on the same data, enriched scalars (the scalar features plus 15 hand-crafted summaries of the tensor, 202 matches, mean ± SD across seeds)")
    OUT.write_text("\n".join(lines), encoding="utf8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
