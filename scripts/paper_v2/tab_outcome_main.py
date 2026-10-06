#!/usr/bin/env python3
"""
表 tab_outcome_main  七个结果类任务上单任务模型（薄、加厚标量）和统一模型的 AUC
============================
每格是跨种子 AUC 均值和标准差，全部取 u1_clean_summary.md。单任务薄标量取“同数据单任务模型（202 场）”表，
加厚标量取“同数据单任务模型，标量加厚版”表，统一模型取“运行 u1”和“运行 u1rich”两表的 test_mean、test_sd。
另用逐事件文件重算一遍逐种子 AUC，核对均值和标准差与汇总表一致（不一致就停）。

输入  audit/u1_clean_summary.md，data/cache/u1_clean/*.npz，data/cache/single_clean{,_rich}/preds/*.npz（只用来核对）
输出  outputs/tables/tab_outcome_main.tex，scripts/paper_v2/out/tab_outcome_main_sources.json

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/paper_v2/tab_outcome_main.py

Last modified 2026-10-06（表注措辞按合作者 10-05 修订，末尾加配对自助区间一句，数见 audit/2026-10-06_paired_bootstrap_2425.md）
"""
from scripts.paper_v2.common import (OUTCOME, SRC, TASK_NAME, Registry, md_find, md_row, mean_sd, single_seed_values,
                                     tex, unified_seed_values, write_table)

NAME = "tab_outcome_main"
# (列名, 来源) 来源是 ("single", 表标题前缀, 模型列) 或 ("unified", 运行标题前缀, 运行标签)
COLS = [
    ("XGBoost", ("single", "## 同数据单任务模型（202", "B2_XGB_360", "single_clean")),
    ("MLP", ("single", "## 同数据单任务模型（202", "M2_MLP_360", "single_clean")),
    ("CNN", ("single", "## 同数据单任务模型（202", "M4_CNN_Full", "single_clean")),
    ("XGBoost", ("single", "## 同数据单任务模型，标量加厚", "B2_XGB_360", "single_clean_rich")),
    ("MLP", ("single", "## 同数据单任务模型，标量加厚", "M2_MLP_360", "single_clean_rich")),
    ("CNN", ("single", "## 同数据单任务模型，标量加厚", "M4_CNN_Full", "single_clean_rich")),
    ("TASG", ("single", "## 同数据单任务模型，标量加厚", "G1_Gating", "single_clean_rich")),
    ("Thin", ("unified", "## 运行 u1（", "u1", None)),
    ("Enriched", ("unified", "## 运行 u1rich（", "u1rich", None)),
]


def stack(mean, sd):
    return f"\\makecell{{{tex(mean)}\\\\[-2pt]{{\\scriptsize $\\pm${sd}}}}}"


def main():
    R = Registry(NAME)
    seeds_row = []
    rows = []
    for r, t in enumerate(OUTCOME):
        ln_thin = md_find(SRC["u1"], "## 同数据单任务模型（202", 0, task=t)
        n_test = R.md(f"{t}/n_test", SRC["u1"], ln_thin, "n_test", fmt="int", cell=[r, 1])
        cells = [TASK_NAME[t], f"{int(n_test):,}"]
        for c, (label, (kind, head, key, d)) in enumerate(COLS):
            col = c + 2
            cid = f"{t}/{kind}_{key}" + ("_rich" if d == "single_clean_rich" else "")
            if kind == "single":
                ln = md_find(SRC["u1"], head, 0, task=t)
                m = R.md(cid + "/mean", SRC["u1"], ln, key, "mean", fmt="f4", cell=[r, col])
                s = R.md(cid + "/sd", SRC["u1"], ln, key, "sd", fmt="f4", cell=[r, col])
                n = int(md_row(SRC["u1"], ln)[key].split("n=")[1])
                chk = single_seed_values(d, t, key)
            else:
                ln = md_find(SRC["u1"], head, 0, task=t)
                m = R.md(cid + "/mean", SRC["u1"], ln, "test_mean", fmt="f4", cell=[r, col])
                s = R.md(cid + "/sd", SRC["u1"], ln, "test_sd", fmt="f4", cell=[r, col])
                chk = unified_seed_values(key, t, "metric")
                n = len(chk)
            cm, cs, cn = mean_sd(chk)
            assert round(cm, 4) == round(m, 4) and round(cs, 4) == round(s, 4) and cn == n, \
                f"{cid} 逐事件重算 {cm:.4f}±{cs:.4f} n={cn} 和汇总表 {m}±{s} n={n} 不一致"
            R.notes.append(f"{cid} 逐事件重算 {cm:.4f} ± {cs:.4f}（{cn} 个种子）和汇总表一致")
            if r == 0:
                seeds_row.append(str(n))
            cells.append(stack(f"{m:.4f}", f"{s:.4f}"))
        rows.append(" & ".join(cells) + " \\\\")
    # 种子数登记，单任务取单元格里的 n=，统一模型取运行标题里的种子列表
    for c, (label, (kind, head, key, d)) in enumerate(COLS):
        if kind == "single":
            ln = md_find(SRC["u1"], head, 0, task="pass")
            R.md(f"seeds/{key}_{d}", SRC["u1"], ln, key, "n", fmt="int", cell=["seeds", c + 2])
        else:
            L = SRC["u1"].read_text(encoding="utf8").split("\n")
            hl = next(i for i, x in enumerate(L) if x.startswith(head)) + 1
            R.text(f"seeds/{key}", seeds_row[c], SRC["u1"], hl, "种子 [0, 1, 2, 3]", cell=["seeds", c + 2])

    body = ["\\toprule",
            " & & \\multicolumn{3}{c}{Thin scalars} & \\multicolumn{4}{c}{Enriched scalars} & \\multicolumn{2}{c}{Unified model} \\\\",
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-11}",
            "Task & Events & " + " & ".join(lbl for lbl, _ in COLS) + " \\\\",
            "Seeds & & " + " & ".join(seeds_row) + " \\\\",
            "\\midrule"] + rows + ["\\bottomrule"]
    note = ("Each cell gives the AUC (area under the receiver operating characteristic curve) averaged over seeds and, below it, the standard deviation across seeds; the Seeds row gives their number. "
            "Events is the number of test events in the 202 clean matches of 2024/25. Thin scalars are six event features (ball "
            "position, distance and two angles to goal, under pressure), ten pass features for Pass only, and four 360 summaries (distance "
            "to the nearest defender, defenders goal-side, visible teammates, visible opponents; recomputed from the perspective-corrected frame for Dribble and Tackle). Enriched scalars add 15 counts taken from "
            "the same $7 \\times 8 \\times 12$ tensor (teammates and opponents in the ball cell, in its $3 \\times 3$ and $5 \\times 5$ "
            "neighbourhoods, ahead of and behind the ball, in total, distance to the nearest one, and opponents in the lane to goal). "
            "MLP is a multilayer perceptron on the scalars. CNN, a concatenation network with a convolutional branch, joins the SoccerMap tensor encoding with the scalar encoding. "
            "TASG (task-adaptive spatial gating) mixes the two encodings with a learned gate and is trained for one task. The unified model is trained on all eight tasks "
            "with the pooled spatial encoder. For the enriched unified model, the criterion written in advance required at least "
            "six of the seven tasks to be no more than 0.003 below enriched XGBoost. It was judged on three seeds, with five of seven tasks "
            "meeting it (not matched); the four seeds reported here give the same result. Differences quoted in the text between the enriched CNN and enriched XGBoost are paired over seeds 0 to 2, shared by both models, and can differ by a few thousandths from the difference between the means shown here, where enriched XGBoost is averaged over five seeds. "
            "Over 2,000 bootstrap resamples of the 202 test matches, the 95\\% interval of the enriched CNN minus enriched XGBoost (paired over seeds 0 to 2) excludes zero only on pass ($[-0.0083, -0.0056]$), and that of the thin unified model minus enriched XGBoost (four against five seeds) excludes zero only on pass ($[-0.0051, -0.0027]$) and pressure ($[0.0007, 0.0099]$); the intervals for all tasks are given in the Supplementary Material.")
    caption = ("AUC of the single-task models with thin and enriched scalars and of the unified model on the seven outcome tasks, "
               "202 test matches of 2024/25.")
    write_table(NAME, body, caption, "tab:outcome_main", env="table*", note=note, colspec="l r " + "c " * len(COLS))
    R.save()
    print(f"[{NAME}] 写好 {len(R.items)} 个数")


if __name__ == "__main__":
    main()
