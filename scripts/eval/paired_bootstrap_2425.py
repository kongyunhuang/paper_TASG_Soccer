#!/usr/bin/env python3
"""
paired_bootstrap_2425.py, match bootstrap intervals for the 2024/25 outcome-task comparisons
============================================================================================
Resamples the 202 clean test matches of 2024/25 and gives 95% intervals for seed-mean AUC differences to enriched
XGBoost on the seven outcome tasks.
The procedure is that of holdout_2526_summarize.py (AUC from AUCFast with integer match weights; one resample shared by
all tasks and models; RandomState 20261003; 2.5 and 97.5 percentiles). Differences are taken between seed means with
the pairing rule of its paired_diff. Single-task models are paired over their shared seeds (a seed is the same data
split for both), and the unified model against a single-task model is a difference of seed means, not paired.
  A   thin unified model u1 (seeds 0 to 3) minus enriched XGBoost (seeds 0 to 4), not paired
  B   enriched CNN (seeds 0 to 2) minus enriched XGBoost, paired over the shared seeds 0 to 2
  B5  as B, with enriched XGBoost averaged over its five seeds, the difference of the two seed means in Table 7 of the paper
  C   enriched unified model u1rich (seeds 0 to 3) minus enriched XGBoost, not paired
Before resampling, the seed mean of every model and task is checked to four decimals against the provenance registry
of the outcome table (tab_outcome_main), and the script stops on any mismatch. It does not write under data/.

Inputs   data/cache/u1_clean/{u1,u1rich}_s*.npz
         data/cache/single_clean_rich/preds/{task}_{model}_s*.npz and {task}_test_meta.npz
         data/raw_scan/match_state_2425.parquet (202 matches through summarize_u1_clean.clean_matches)
         outputs/provenance/tab_outcome_main_sources.json (written by scripts/paper/tab_outcome_main.py)
Outputs  results/paired_bootstrap_2425.json and results/paired_bootstrap_2425.md

Usage
  PYTHONPATH=. python -u scripts/eval/paired_bootstrap_2425.py --n_boot 2000

Last modified 2026-10-28
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
REGISTRY = Path("outputs/provenance/tab_outcome_main_sources.json")
OUT_JSON = Path("results/paired_bootstrap_2425.json")
OUT_MD = Path("results/paired_bootstrap_2425.md")
TASKS8 = ["pass", "dest", "shot", "interception", "ball_recovery", "pressure", "dribble", "tackle"]
TASKS7 = ["pass", "shot", "interception", "ball_recovery", "pressure", "dribble", "tackle"]
MEAN3 = ["pass", "ball_recovery", "pressure"]
TASK_NAME = {"pass": "Pass", "shot": "Shot", "interception": "Interception", "ball_recovery": "Ball recovery",
             "pressure": "Pressure", "dribble": "Dribble", "tackle": "Tackle",
             "mean7": "Mean of seven tasks", "mean3": "Mean of three tasks (Pass, Ball recovery, Pressure)"}
MODELS = {  # name: (source, model name or tag in the files, seeds)
    "u1": ("unified", "u1", [0, 1, 2, 3]),
    "u1rich": ("unified", "u1rich", [0, 1, 2, 3]),
    "xgbrich": ("single", "B2_XGB_360", [0, 1, 2, 3, 4]),
    "cnnrich": ("single", "M4_CNN_Full", [0, 1, 2]),
}
REG_ID = {"u1": "unified_u1", "u1rich": "unified_u1rich", "xgbrich": "single_B2_XGB_360_rich", "cnnrich": "single_M4_CNN_Full_rich"}
# (key, left model, left seeds, right model, right seeds, description)
CONTRASTS = [
    ("A_u1_minus_xgbrich", "u1", [0, 1, 2, 3], "xgbrich", [0, 1, 2, 3, 4], "thin unified model (mean of four seeds) minus enriched XGBoost (mean of five seeds), not paired"),
    ("B_cnnrich_minus_xgbrich_paired3", "cnnrich", [0, 1, 2], "xgbrich", [0, 1, 2], "enriched CNN minus enriched XGBoost, paired over seeds 0 to 2, equal to the difference of the two three-seed means"),
    ("B5_cnnrich_minus_xgbrich_xgb5", "cnnrich", [0, 1, 2], "xgbrich", [0, 1, 2, 3, 4], "enriched CNN (mean of three seeds) minus enriched XGBoost (mean of five seeds), the difference of the two means in Table 7 of the paper"),
    ("C_u1rich_minus_xgbrich", "u1rich", [0, 1, 2, 3], "xgbrich", [0, 1, 2, 3, 4], "enriched unified model (mean of four seeds) minus enriched XGBoost (mean of five seeds), not paired"),
]


def load_runs():
    """Returns {(model name, seed, task): dict(score, y, match)}, restricted to the 202 clean matches"""
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
    """{(model name, task): {seed: AUC}}; w_of(run) gives the weights, None means unweighted"""
    out = {}
    for (name, sd, t), r in runs.items():
        w = w_of(r)
        out.setdefault((name, t), {})[sd] = r["fast"].auc(np.ones(len(r["y"])) if w is None else w)
    return out


def contrasts(per):
    """Seed-mean difference of every contrast on every task, plus the mean over seven tasks and over three tasks (Pass, Ball recovery, Pressure)"""
    out = {}
    for key, a, sa, b, sb, _ in CONTRASTS:
        d = {t: float(np.mean([per[(a, t)][s] for s in sa]) - np.mean([per[(b, t)][s] for s in sb])) for t in TASKS7}
        d["mean7"] = float(np.mean([d[t] for t in TASKS7]))
        d["mean3"] = float(np.mean([d[t] for t in MEAN3]))
        out[key] = d
    return out


def per_seed_paired(per):
    """Per-seed differences of contrast B"""
    return {t: {s: per[("cnnrich", t)][s] - per[("xgbrich", t)][s] for s in [0, 1, 2]} for t in TASKS7}


def check_against_table7(per):
    reg = json.load(open(REGISTRY, encoding="utf8"))
    val = {it["id"]: it["value"] for it in reg["items"]}
    lines = []
    for name in MODELS:
        for t in TASKS7:
            m = float(np.mean(list(per[(name, t)].values())))
            ref = val[f"{t}/{REG_ID[name]}/mean"]
            ok = round(m, 4) == round(ref, 4)
            lines.append((name, t, m, ref, ok))
            if not ok:
                raise SystemExit(f"[check] {name} {t} recomputed {m:.6f} differs from the outcome table registry {ref} at four decimals, stopping")
    return lines


def write_md(out):
    """Markdown report of the intervals in out (the dictionary that is also written as json)"""
    res, n_ev = out["results"], out["n_events"]
    sgn = lambda v: f"{v:+.4f}".replace("+0.0000", "0.0000").replace("-0.0000", "0.0000")
    pln = lambda v: f"{v:.4f}".replace("-0.0000", "0.0000")
    ci = lambda r, f: f"[{f(r['lo'])}, {f(r['hi'])}]"
    has0 = lambda r: "yes" if r["lo"] <= 0 <= r["hi"] else "no"
    ev = lambda t: f"{n_ev[t]:,}" if t in n_ev else ""
    L = [f"# Match bootstrap intervals for the 2024/25 outcome-task comparisons ({out['n_matches']} matches, {out['n_boot']:,} resamples)", "",
         "Generated by `scripts/eval/paired_bootstrap_2425.py`. In each resample the test matches are drawn with replacement, once for all tasks, models and seeds "
         f"(RandomState {out['boot_seed']}), and AUC is recomputed with the match counts as weights (AUCFast of `scripts/eval/holdout_2526_summarize.py`, ties count one half). "
         "Every resample first gives the AUC of each model and seed, then the seed means below, then their difference. Intervals are the 2.5 and 97.5 percentiles over the resamples. "
         "The share of resamples above zero is given for reference only. Point estimates and intervals are AUC differences.", "",
         "## Seeds", "", "| Contrast | Left | Right | Pairing |", "|---|---|---|---|"]
    for key, a, sa, b, sb, desc in CONTRASTS:
        L.append(f"| {key} | {a} seeds {', '.join(map(str, sa))} | {b} seeds {', '.join(map(str, sb))} | {desc} |")
    L.append("")
    for key, title, rows, cols in [
            ("A_u1_minus_xgbrich", "A, thin unified model (four seeds) minus enriched XGBoost (five seeds)", TASKS7 + ["mean7", "mean3"], "full"),
            ("B_cnnrich_minus_xgbrich_paired3", "B, enriched CNN minus enriched XGBoost, paired over seeds 0 to 2", TASKS7 + ["mean7", "mean3"], "full"),
            ("B5_cnnrich_minus_xgbrich_xgb5", "B5, enriched CNN (three seeds) minus enriched XGBoost (five seeds)", TASKS7 + ["mean7"], "short"),
            ("C_u1rich_minus_xgbrich", "C, enriched unified model (four seeds) minus enriched XGBoost (five seeds)", TASKS7 + ["mean7"], "contains")]:
        L += [f"## {title}", ""]
        if cols == "full":
            L += ["| Task | Events | Point estimate | 95% CI | CI contains 0 | Share of resamples above zero |", "|---|---:|---:|---|---|---:|"]
            L += [f"| {TASK_NAME[t]} | {ev(t)} | {sgn(res[key][t]['point'])} | {ci(res[key][t], sgn)} | {has0(res[key][t])} | {res[key][t]['share_pos']:.3f} |" for t in rows]
        elif cols == "short":
            L += ["| Task | Point estimate | 95% CI |", "|---|---:|---|"]
            L += [f"| {TASK_NAME[t]} | {sgn(res[key][t]['point'])} | {ci(res[key][t], sgn)} |" for t in rows]
        else:
            L += ["| Task | Point estimate | 95% CI | CI contains 0 |", "|---|---:|---|---|"]
            L += [f"| {TASK_NAME[t]} | {sgn(res[key][t]['point'])} | {ci(res[key][t], sgn)} | {has0(res[key][t])} |" for t in rows]
        L.append("")
        if key.startswith("B_"):
            L += ["Per-seed differences of B on the 202 matches, without resampling", "", "| Task | Seed 0 | Seed 1 | Seed 2 |", "|---|---:|---:|---:|"]
            L += [f"| {TASK_NAME[t]} | " + " | ".join(sgn(out["per_seed_B"][t][s]) for s in ("0", "1", "2")) + " |" for t in TASKS7]
            L.append("")
    a, b = "A_u1_minus_xgbrich", "B_cnnrich_minus_xgbrich_paired3"
    L += ["## Supplementary table", "",
          "Each cell gives the seed-mean AUC difference to enriched XGBoost and, in brackets, the 95% match bootstrap interval. Point estimates use unrounded seed means.", "",
          "| Task | Events | Enriched CNN minus enriched XGBoost, seeds 0 to 2 paired | Thin unified model (4 seeds) minus enriched XGBoost (5 seeds) |", "|---|---:|---|---|"]
    L += [f"| {TASK_NAME[t]} | {ev(t)} | {pln(res[b][t]['point'])} {ci(res[b][t], pln)} | {pln(res[a][t]['point'])} {ci(res[a][t], pln)} |" for t in TASKS7]
    L.append("")
    OUT_MD.write_text("\n".join(L), encoding="utf8")


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
    print(f"[load] {len(runs)} model x seed x task runs, {len(matches)} matches  ({time.time() - t0:.0f}s)", flush=True)
    assert len(matches) == 202, len(matches)
    n_ev = {t: len(runs[("xgbrich", 0, t)]["y"]) for t in TASKS7}
    # the unified and single-task models must score the same test events; compare event and positive counts per match
    for (name, sd, t), r in runs.items():
        ref = runs[("xgbrich", 0, t)]
        same_n = (np.bincount(r["_mi"], minlength=len(matches)) == np.bincount(ref["_mi"], minlength=len(matches))).all()
        same_pos = (np.bincount(r["_mi"], r["y"], minlength=len(matches)) == np.bincount(ref["_mi"], ref["y"], minlength=len(matches))).all()
        assert same_n and same_pos, f"{name} s{sd} {t} events or positives per match differ from enriched XGBoost"

    per = seed_means(runs, lambda r: None)
    chk = check_against_table7(per)
    print(f"[check] seed means of four models on seven tasks equal the outcome table registry to four decimals ({len(chk)} values)", flush=True)
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
            print(f"[boot] {b + 1}/{a.n_boot}  {el:.0f}s  about {el / (b + 1) * (a.n_boot - b - 1):.0f}s left", flush=True)

    res = {}
    for k, *_ in CONTRASTS:
        res[k] = {}
        for t in TASKS7 + ["mean7", "mean3"]:
            v = np.array(boots[k][t])
            res[k][t] = dict(point=point[k][t], lo=float(np.nanpercentile(v, 2.5)), hi=float(np.nanpercentile(v, 97.5)),
                             share_pos=float((v > 0).mean()), n_nan=int(np.isnan(v).sum()))
    out = dict(n_boot=a.n_boot, boot_seed=BOOT_SEED, n_matches=int(len(matches)), n_events=n_ev,
               contrasts={k: dict(desc=desc, left=lft, left_seeds=sa, right=rgt, right_seeds=sb) for k, lft, sa, rgt, sb, desc in CONTRASTS},
               results=res, per_seed_B={t: {str(s): v for s, v in d.items()} for t, d in seeds_b.items()},
               seed_means={f"{n}|{t}": {str(s): v for s, v in d.items()} for (n, t), d in per.items()})
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8")
    write_md(out)
    for k, *_ in CONTRASTS:
        print(f"[res] {k}", flush=True)
        for t in TASKS7 + ["mean7", "mean3"]:
            r = res[k][t]
            print(f"      {t:14s} {r['point']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  share above zero {r['share_pos']:.3f}", flush=True)
    print(f"[done] {OUT_JSON} and {OUT_MD}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
