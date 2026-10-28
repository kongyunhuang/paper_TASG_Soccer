#!/usr/bin/env python3
"""
tab_destination.py, table of the pass destination task
======================================================
Top-1 and skill on the 96-cell destination task for the strongest rule, the scalar models, the CNN, the gated models,
the unified model and the resolution-preserving variant.
On the 202 matches of 2024/25 the rule comes from results/dest_rule_baselines.md. The single-task models come from the
two single-task tables of results/u1_clean_summary.md (skill mean and SD, top-1 mean in parentheses); the summary has no
SD of top-1 across seeds, so it is recomputed from the per-event files, together with a check that the recomputed mean
equals the summary. The unified models come from the tables of runs u1 and u1fcn, with the SD of skill also recomputed
from the per-event files. On the 146 matches of 2025/26 the rule comes from results/holdout_2526_dest_rules.json and the
models from the metrics of results/holdout_2526_summary.json.

Inputs   results/dest_rule_baselines.md, results/u1_clean_summary.md, results/holdout_2526_summary.json,
         results/holdout_2526_dest_rules.json, data/cache/u1_clean/*.npz, data/cache/single_clean{,_rich}/preds/dest_*.npz
Outputs  outputs/tables/tab_destination.tex, outputs/provenance/tab_destination_sources.json

Usage
  PYTHONPATH=. python -u scripts/paper/tab_destination.py

Last modified 2026-10-28
"""
from scripts.paper.common import (SRC, Registry, json_value, md_find, md_row, mean_sd, single_seed_values, tex,
                                  unified_seed_values, write_table)

NAME = "tab_destination"
RULE = "Table times teammates over opponents (a=2, b=0.5)"
ROWS = [  # (row label, kind, single-task directory or run tag, model)
    ("Strongest rule", "rule", None, None),
    ("MLP, thin scalars", "single", "single_clean", "M2_MLP_360"),
    ("MLP, enriched scalars", "single", "single_clean_rich", "M2_MLP_360"),
    ("CNN, thin scalars", "single", "single_clean", "M4_CNN_Full"),
    ("CNN, enriched scalars", "single", "single_clean_rich", "M4_CNN_Full"),
    ("TASG, thin scalars", "single", "single_clean", "G1_Gating"),
    ("TASG, enriched scalars", "single", "single_clean_rich", "G1_Gating"),
    ("Unified model, pooled encoder", "unified", "u1", None),
    ("Resolution-preserving variant", "unified", "u1fcn", None),
]
HEAD = {"single_clean": "## Single-task models on the same data (202", "single_clean_rich": "## Single-task models on the same data, enriched scalars"}
HKEY = {"single_clean": "single", "single_clean_rich": "single_rich"}


def stack(mean, sd):
    return f"\\makecell{{{mean}\\\\[-2pt]{{\\scriptsize $\\pm${sd}}}}}"


def main():
    R = Registry(NAME)
    hj = SRC["hold_json"]
    rows = []
    for r, (label, kind, d, model) in enumerate(ROWS):
        cells = [label]
        if kind == "rule":
            L = SRC["dest_rules"].read_text(encoding="utf8").split("\n")
            ln = next(i for i, x in enumerate(L) if x.startswith(f"| {RULE}")) + 1
            cells.append("n/a")
            t1 = R.md("rule/2425/top1", SRC["dest_rules"], ln, "top1", fmt="f4", cell=[r, 2])
            sk = R.md("rule/2425/skill", SRC["dest_rules"], ln, "skill", fmt="f4", cell=[r, 3])
            R.md("rule/2425/top3", SRC["dest_rules"], ln, "top3", fmt="f4", cell=["note", 0])
            h1 = R.js("rule/2526/top1", SRC["hold_dest_rules"], [RULE, "top1"], fmt="f4", cell=[r, 4])
            hs = R.js("rule/2526/skill", SRC["hold_dest_rules"], [RULE, "skill"], fmt="f4", cell=[r, 5])
            R.js("rule/2526/top3", SRC["hold_dest_rules"], [RULE, "top3"], fmt="f4", cell=["note", 0])
            cells += [f"{t1:.4f}", f"{sk:.4f}", f"{h1:.4f}", f"{hs:.4f}"]
        elif kind == "single":
            ln = md_find(SRC["u1"], HEAD[d], 0, task="dest")
            cid = f"{d}/{model}"
            n = int(md_row(SRC["u1"], ln)[model].split("n=")[1])
            t1 = R.md(cid + "/2425/top1", SRC["u1"], ln, model, "paren", fmt="f4", cell=[r, 2])
            per = single_seed_values(d, "dest", model, "metric")
            pm, ps, pn = mean_sd(per)
            assert round(pm, 4) == round(t1, 4) and pn == n, f"{cid} recomputed top-1 {pm:.4f} n={pn} differs from the summary table {t1} n={n}"
            R.perevent(cid + "/2425/top1_sd", ps, f"top1 of data/cache/{d}/preds/dest_{model}_s*.npz against y of dest_test_meta.npz, 202 matches, "
                       f"sample SD of the per-seed top-1 ({pn} seeds, recomputed mean {pm:.4f} equals the summary table)", fmt="f4", cell=[r, 2])
            sk = R.md(cid + "/2425/skill", SRC["u1"], ln, model, "mean", fmt="f4", cell=[r, 3])
            ss = R.md(cid + "/2425/skill_sd", SRC["u1"], ln, model, "sd", fmt="f4", cell=[r, 3])
            cells += [str(n), stack(f"{t1:.4f}", f"{ps:.4f}"), stack(f"{sk:.4f}", f"{ss:.4f}")]
            key = f"{HKEY[d]}|{model}|dest"
            if key in json_value(hj, ["metrics"]):
                hn = json_value(hj, ["metrics", key, "top1", "n"])
                assert hn == n, f"{key} has {hn} seeds on the held-out set and {n} in 2024/25"
                a = R.js(cid + "/2526/top1", hj, ["metrics", key, "top1", "mean"], fmt="f4", cell=[r, 4])
                b = R.js(cid + "/2526/top1_sd", hj, ["metrics", key, "top1", "sd"], fmt="f4", cell=[r, 4])
                c = R.js(cid + "/2526/skill", hj, ["metrics", key, "skill", "mean"], fmt="f4", cell=[r, 5])
                e = R.js(cid + "/2526/skill_sd", hj, ["metrics", key, "skill", "sd"], fmt="f4", cell=[r, 5])
                cells += [stack(f"{a:.4f}", f"{b:.4f}"), stack(f"{c:.4f}", f"{e:.4f}")]
            else:
                R.notes.append(f"{key} is not in the metrics of the held-out summary json, shown as n.e.")
                cells += ["n.e.", "n.e."]
        else:
            ln = md_find(SRC["u1"], f"## Run {d} (", 0, task="dest")
            t1 = R.md(f"{d}/2425/top1", SRC["u1"], ln, "test_mean", fmt="f4", cell=[r, 2])
            ts = R.md(f"{d}/2425/top1_sd", SRC["u1"], ln, "test_sd", fmt="f4", cell=[r, 2])
            sk = R.md(f"{d}/2425/skill", SRC["u1"], ln, "dest_skill_mean", fmt="f4", cell=[r, 3])
            per = unified_seed_values(d, "dest", "skill")
            pm, ps, pn = mean_sd(per)
            assert round(pm, 4) == round(sk, 4), f"{d} recomputed skill {pm:.4f} differs from the summary table {sk}"
            R.perevent(f"{d}/2425/skill_sd", ps, f"dest_logp_true of data/cache/u1_clean/{d}_s*.npz, 202 matches, sample SD of the per-seed skill "
                       f"({pn} seeds, recomputed mean {pm:.4f} equals the summary table), recomputed with summarize_u1_clean.load_runs", fmt="f4", cell=[r, 3])
            key = f"unified|{d}|dest"
            hn = json_value(hj, ["metrics", key, "top1", "n"])
            assert hn == pn, f"{key} has {hn} seeds on the held-out set and {pn} in 2024/25"
            a = R.js(f"{d}/2526/top1", hj, ["metrics", key, "top1", "mean"], fmt="f4", cell=[r, 4])
            b = R.js(f"{d}/2526/top1_sd", hj, ["metrics", key, "top1", "sd"], fmt="f4", cell=[r, 4])
            c = R.js(f"{d}/2526/skill", hj, ["metrics", key, "skill", "mean"], fmt="f4", cell=[r, 5])
            e = R.js(f"{d}/2526/skill_sd", hj, ["metrics", key, "skill", "sd"], fmt="f4", cell=[r, 5])
            cells += [str(pn), stack(f"{t1:.4f}", f"{ts:.4f}"), stack(f"{sk:.4f}", f"{ps:.4f}"),
                      stack(f"{a:.4f}", f"{b:.4f}"), stack(f"{c:.4f}", f"{e:.4f}")]
        rows.append(" & ".join(cells) + " \\\\")
        if r == 0:
            rows.append("\\addlinespace")
        if r == 6:
            rows.append("\\addlinespace")

    top3_a = next(it["value"] for it in R.items if it["id"] == "rule/2425/top3")
    top3_b = next(it["value"] for it in R.items if it["id"] == "rule/2526/top3")
    body = ["\\toprule",
            " & & \\multicolumn{2}{c}{Test 2024/25, 202 matches} & \\multicolumn{2}{c}{Held out 2025/26, 146 matches} \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
            "Method & Seeds & Top-1 & Skill & Top-1 & Skill \\\\",
            "\\midrule"] + rows + ["\\bottomrule"]
    note = ("The destination is the cell of the pass end location on the $8 \\times 12$ grid (96 classes); 26,634 test passes in 2024/25 "
            "and 125,130 in 2025/26. Top-1 is the share of passes whose true cell has the highest predicted probability. Skill is "
            "$1 - \\ell / \\ln 96$, where $\\ell$ is the mean log loss, so a uniform guess scores 0. Learned models show the mean over seeds "
            "and, below it, the standard deviation across seeds. The strongest rule scores each cell by the destination distribution of the "
            "start cell in the training seasons, multiplied by $(1 + 2m)$ and divided by $(1 + 0.5o)$, where $m$ and $o$ are the teammates "
            "and opponents in the cell; the two weights were chosen on held-out training matches. It is deterministic (no seeds, n/a) and "
            f"is the strongest of five rules on both test sets. Its top-3 accuracy is {top3_a:.4f} in 2024/25 and {top3_b:.4f} in 2025/26; "
            "top-3 accuracy was not stored for the learned models. MLP (multilayer perceptron), CNN (concatenation network with a convolutional branch) and TASG (task-adaptive spatial gating) are single-task models; thin scalars are six event features and "
            "four 360 summary features (the ten pass dimensions are zero for this task), and enriched scalars add 15 counts taken from the same tensor. The single-task TASG models were not evaluated on 2025/26 (n.e.). The resolution-preserving variant is the "
            "unified model with a spatial encoder that keeps the $8 \\times 12$ resolution and scores every cell; under the encoder rule "
            "written in advance, it is a supplementary result, and the unified model with the pooled encoder is the main model.")
    caption = ("Pass destination on the 202 test matches of 2024/25 and the 146 held-out matches of 2025/26.")
    write_table(NAME, body, caption, "tab:destination", env="table", note=note, colspec="l c c c c c")
    R.save()
    print(f"[{NAME}] {len(R.items)} numbers written")


if __name__ == "__main__":
    main()
