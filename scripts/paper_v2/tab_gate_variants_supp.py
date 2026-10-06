#!/usr/bin/env python3
"""
补充材料表 tab_gate_variants_supp  门控变体和对照，每个预先写下标准的实验一行，写设置、问题、成立条件、结果和判定
============================
数取 u1_clean_summary.md 里对应的结构化小节（消融对照、加厚标量对基准、门控代价确认步、硬门控探索步、拼接对照、干预），
加厚标量成绩一问的三个种子数取它自己的标准文件末尾的结果表；判定词逐条取七份“预先写下”文件末尾的判定句，照搬翻译，不改口径。

输入  audit/u1_clean_summary.md，memory_logs/2026-10-02_*预先写下*.md（两通道、加厚标量、门控代价、硬门控、拼接对照、主模型编码器）
输出  outputs/tables/tab_gate_variants_supp.tex，scripts/paper_v2/out/tab_gate_variants_supp_sources.json

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/paper_v2/tab_gate_variants_supp.py

Last modified 2026-10-04
"""
from scripts.paper_v2.common import SRC, Registry, fmt_num, md_find, md_value, tex, write_table

NAME = "tab_gate_variants_supp"
U = SRC["u1"]


def main():
    R = Registry(NAME)
    rows = []

    def lst(vals, fmt):
        return ", ".join(tex(fmt_num(v, fmt)) for v in vals)

    # 1 两通道
    r = 0
    sp, pr = [], []
    for sd in [0, 1, 2]:
        ln = md_find(U, "## 门控均值对球员通道消融下降量", 0, run="u1_2ch", seed=sd)
        sp.append(R.md(f"2ch/s{sd}/spearman", U, ln, "spearman_gate_vs_player_drop", fmt="f3", cell=[r, 4]))
        pr.append(R.md(f"2ch/s{sd}/pressure_rank", U, ln, "pressure_gate_rank", fmt="int", cell=[r, 4]))
    v1 = R.text("2ch/verdict", "Inconclusive", SRC["crit_2ch"], 31, "**判定，不确定。**", cell=[r, 5])
    rows.append(["Two-channel spatial input (players only)", "0, 1, 2",
                 "Does the gate follow how much each task loses when player positions are shuffled?",
                 "Spearman between gate and that loss $\\geq$ 0.70 and Pressure gate in the top four, in every seed",
                 f"Spearman {lst(sp, 'f3')}; Pressure gate rank {', '.join(str(int(x)) for x in pr)}", v1])

    # 2 加厚标量，门控一问
    r = 1
    dr, td, dc = [], [], []
    for sd in [0, 1, 2]:
        ln = md_find(U, "## 加厚标量的统一模型（u1rich）对同种子基准", 0, seed=sd)
        dr.append(R.md(f"rich_gate/s{sd}/drop", U, ln, "outcome_gate_avg_drop", fmt="f3", cell=[r, 4]))
        td.append(R.mdre(f"rich_gate/s{sd}/tasks_down", U, ln, "tasks_down", r"^(\d+)/7", fmt="int", cell=[r, 4]))
        dc.append(R.md(f"rich_gate/s{sd}/dest_change", U, ln, "dest_gate_change", fmt="f3", cell=[r, 4]))
    v2 = R.text("rich_gate/verdict", "Inconclusive; the expected effect was not seen", SRC["crit_rich"], 50, "**判定，不确定，实质上没有看到预想的效应。**", cell=[r, 5])
    rows.append(["Enriched scalars in the unified model, gate", "0, 1, 2",
                 "With enriched scalars, do the outcome-task gates close while the destination gate stays?",
                 "Mean outcome gate falls by $\\geq$ 0.10, at least 5 of 7 outcome gates fall, destination gate changes by $<$ 0.10 and stays first, in every seed",
                 f"Fall {lst(dr, 'f3')}; outcome gates falling {', '.join(str(int(x)) for x in td)} of 7; destination gate change {lst(dc, 'f3')}", v2])

    # 3 加厚标量，成绩一问
    r = 2
    lp = md_find(SRC["crit_rich"], "### 成绩这一问", 0, **{"任务": "Pass"})
    ls = md_find(SRC["crit_rich"], "### 成绩这一问", 0, **{"任务": "Shot"})
    p3 = R.mdre("rich_perf/3seeds/pass", SRC["crit_rich"], lp, "u1rich 减加厚 XGBoost", r"负 ([0-9.]+)", fmt="f4", cell=[r, 4], sign=-1)
    s3 = R.mdre("rich_perf/3seeds/shot", SRC["crit_rich"], ls, "u1rich 减加厚 XGBoost", r"负 ([0-9.]+)", fmt="f4", cell=[r, 4], sign=-1)
    l4p = md_find(U, "## 加厚标量的统一模型（u1rich）对同种子基准", 1, task="pass")
    l4s = md_find(U, "## 加厚标量的统一模型（u1rich）对同种子基准", 1, task="shot")
    p4 = R.md("rich_perf/4seeds/pass", U, l4p, "rich_minus_rich_xgb", fmt="f4", cell=[r, 4])
    s4 = R.md("rich_perf/4seeds/shot", U, l4s, "rich_minus_rich_xgb", fmt="f4", cell=[r, 4])
    nok = R.mdre("rich_perf/n_ok", U, md_find(U, "## 加厚标量的统一模型（u1rich）对同种子基准", 1, task="tackle") + 4, None,
           r"只有 (\d)/7 个任务的差不低于负 0.003", fmt="int", cell=[r, 4])
    v3 = R.text("rich_perf/verdict", "Not matched", SRC["crit_rich"], 63, "**判定，没追平。**", cell=[r, 5])
    rows.append(["Enriched scalars in the unified model, AUC", "0, 1, 2 (also 3)",
                 "Does the enriched unified model match enriched XGBoost on the outcome tasks?",
                 "At least 6 of 7 outcome tasks no more than 0.003 below enriched XGBoost",
                 f"{int(nok)} of 7; Pass {tex(fmt_num(p3, 'f4'))}, Shot {tex(fmt_num(s3, 'f4'))} with three seeds; Pass {tex(fmt_num(p4, 'f4'))}, "
                 f"Shot {tex(fmt_num(s4, 'f4'))} and still 5 of 7 with four seeds", v3])

    # 4 软代价，加厚
    r = 3
    og, tw, da = [], [], []
    for sd in [1, 2, 3]:
        ln = md_find(U, "### 软门控，标量加厚，确认步", 0, seed=sd)
        og.append(R.md(f"pen_rich/s{sd}/outcome_gate", U, ln, "outcome_gate_avg", fmt="f3", cell=[r, 4]))
        tw.append(R.mdre(f"pen_rich/s{sd}/within", U, ln, "tasks_within_tolerance", r"^(\d)/8", fmt="int", cell=[r, 4]))
        da.append(R.md(f"pen_rich/s{sd}/drop_all", U, ln, "drop_all_outcome_avg", fmt="f3", cell=[r, 4]))
    v4 = R.text("pen_rich/verdict", "Inconclusive", SRC["crit_pen"], 110, "判定 **不确定**", cell=[r, 5])
    rows.append(["Soft gate with an opening penalty ($\\lambda = 0.01$), enriched scalars", "1, 2, 3",
                 "With a price on opening and enriched scalars, do the outcome gates close without loss of AUC?",
                 "Mean outcome gate $\\leq$ 0.10 with the destination gate $\\geq$ 0.50 and first; at least 7 of 8 tasks within tolerance; "
                 "AUC loss of the outcome tasks under a full tensor shuffle $\\leq$ 0.01; in every seed",
                 f"Mean outcome gate {lst(og, 'f3')}; tasks within tolerance {', '.join(str(int(x)) for x in tw)} of 8; loss under shuffle {lst(da, 'f3')}", v4])

    # 5 软代价，薄
    r = 4
    s8, s7, tw2 = [], [], []
    for sd in [1, 2, 3]:
        ln = md_find(U, "### 软门控，标量薄，确认步", 0, seed=sd)
        s8.append(R.md(f"pen_thin/s{sd}/spearman8", U, ln, "spearman_8", fmt="f2", cell=[r, 4]))
        s7.append(R.md(f"pen_thin/s{sd}/spearman7", U, ln, "spearman_7", fmt="f2", cell=[r, 4]))
        tw2.append(R.mdre(f"pen_thin/s{sd}/within", U, ln, "tasks_within_tolerance", r"^(\d)/8", fmt="int", cell=[r, 4]))
    v5 = R.text("pen_thin/verdict", "Not supported", SRC["crit_pen"], 118, "判定 **不成立**", cell=[r, 5])
    rows.append(["Soft gate with an opening penalty ($\\lambda = 0.01$), thin scalars", "1, 2, 3",
                 "With a price on opening, does the gate order the tasks by their single-task CNN gain?",
                 "Spearman with that gain $\\geq$ 0.70 over eight tasks and $\\geq$ 0.60 over the seven AUC tasks; at least 7 of 8 tasks within tolerance; in every seed",
                 f"Spearman {lst(s8, 'f2')} (eight tasks) and {lst(s7, 'f2')} (seven); tasks within tolerance {', '.join(str(int(x)) for x in tw2)} of 8", v5])

    # 6、7 硬门控
    for r, (key, head, crit_line, crit_txt, lab) in enumerate([
            ("hard_rich", "### 硬门控，标量加厚，探索步", 81, "甲二记为没有成立", "enriched scalars"),
            ("hard_thin", "### 硬门控，标量薄，探索步", 90, "乙二记为没有成立", "thin scalars")], start=5):
        vb = R.md(f"{key}/val_base", U, md_find(U, head, 0, lam="0"), "val_score", fmt="f4", cell=[r, 4])
        v01 = R.md(f"{key}/val_001", U, md_find(U, head, 0, lam="0.01"), "val_score", fmt="f4", cell=[r, 4])
        v03 = R.md(f"{key}/val_003", U, md_find(U, head, 0, lam="0.03"), "val_score", fmt="f4", cell=[r, 4])
        vv = R.text(f"{key}/verdict", "Not established; stopped at the $\\lambda$ selection step", SRC["crit_hard"], crit_line, crit_txt, cell=[r, 5])
        rows.append([f"Hard gate with an opening penalty, {lab}", "0 (selection)",
                     "Does a binary gate, which cannot pass a small weight on, give a gate reading with meaning?",
                     "A $\\lambda$ enters the confirmation step only if its validation score is within 0.002 of the model without penalty",
                     f"Validation score {v01:.4f} ($\\lambda = 0.01$) and {v03:.4f} ($\\lambda = 0.03$) against {vb:.4f}; no $\\lambda$ kept", vv])

    # 8 拼接对照
    r = 7
    ex = []
    for sd in [0, 1, 2]:
        lg = md_find(U, "## 落点干预下其余任务被拖累多少", 0, seed=sd, fusion="门控")
        lc = md_find(U, "## 落点干预下其余任务被拖累多少", 0, seed=sd, fusion="拼接")
        g, c = md_value(U, lg, "other7_avg_change"), md_value(U, lc, "other7_avg_change")
        ex.append(R.derived(f"concat/s{sd}/extra_loss", round(g - c, 4), f"L{lg} 门控 other7_avg_change 减 L{lc} 拼接 other7_avg_change",
                            [f"md:{lg}", f"md:{lc}"], fmt="f4", cell=[r, 4]))
    R.mdre("concat/summary_line", U, md_find(U, "## 落点干预下其余任务被拖累多少", 0, seed=2, fusion="拼接") + 4, None,
           r"逐种子） \[([-0-9.]+),", fmt="f4", cell=None)
    v8 = R.text("concat/verdict", "Not supported", SRC["crit_concat"], 39, "**判定，不成立。**", cell=[r, 5])
    rows.append(["Concatenation in place of the gate, destination shuffled", "0, 1, 2",
                 "Under the destination intervention, does the gate protect the other tasks better than concatenation?",
                 "With concatenation the other seven tasks lose at least 0.005 more AUC on average than with the gate, in every seed",
                 f"Extra loss with concatenation {lst(ex, 'f4')}", v8])

    # 9 保留分辨率编码器
    r = 8
    li = md_find(U, "## 干预", 0, run="u1fcn_shuf_dest", seed=0, task="dest")
    gch = R.md("fcn/gate_change", U, li, "gate_change", fmt="f3", cell=[r, 4])
    rb = R.md("fcn/rank_base", U, li, "rank_base", fmt="int", cell=[r, 4])
    ra = R.md("fcn/rank_after", U, li, "rank_shuffled_run", fmt="int", cell=[r, 4])
    t1 = [R.mdre(f"fcn/top1_s{k}", SRC["crit_enc"], 40, None, rx, fmt="f1", cell=[r, 4])
          for k, rx in enumerate([r"top-1 是 ([0-9.]+)%", r"top-1 是 [0-9.]+%、([0-9.]+)%", r"top-1 是 [0-9.]+%、[0-9.]+%、([0-9.]+)%"])]
    v9 = R.text("fcn/verdict", "Not met; the pooled encoder is the main model", SRC["crit_enc"], 43, "判定不变，主模型用原编码器", cell=[r, 5])
    rows.append(["Resolution-preserving encoder as the main model", "0, 1, 2 (intervention 0)",
                 "Does the encoder that keeps the $8 \\times 12$ resolution meet the rule written in advance for the main model?",
                 "Destination top-1 above 15.8\\% in every seed with a mean of at least 19\\%; destination intervention lowers its gate by $\\geq$ 0.20 "
                 "into the last two ranks; stable gate ranking",
                 f"Top-1 {t1[0]:.1f}\\%, {t1[1]:.1f}\\%, {t1[2]:.1f}\\%; gate change under the intervention {tex(fmt_num(gch, 'f3'))}, rank {int(rb)} to {int(ra)} (one seed)", v9])

    body = ["\\toprule", "Variant & Seeds & Question & Supported if & Result & Verdict \\\\", "\\midrule"]
    for k, row in enumerate(rows):
        body.append(" & ".join(row) + " \\\\")
        if k < len(rows) - 1:
            body.append("\\addlinespace[2pt]")
    body.append("\\bottomrule")
    note = ("All runs use the unified model with the pooled encoder on the 202 test matches of 2024/25, unless the row states otherwise. "
            "Each criterion was written down before the first result of its experiment, except the encoder rule, which was written when "
            "only the first seed of the resolution-preserving model had a result, before any result of its other seeds or its intervention existed; the verdicts follow those criteria. The other criteria also "
            "defined a condition for not supported and call every other outcome inconclusive, except the AUC question of the enriched "
            "unified model, which has only matched and not matched. "
            "AUC is the area under the receiver operating characteristic curve and CNN the concatenation network with a convolutional branch. Within tolerance means no more than 0.003 below the matching model without penalty (0.005 in top-1 for Destination). "
            "For the penalty experiments, $\\lambda$ was chosen on seed 0 and judged on seeds 1, 2 and 3; a penalised soft gate adds "
            "$\\lambda$ times the mean gate to the loss, and a hard gate takes each fused dimension from one branch only.")
    caption = "Gate variants and controls, each with the criterion written before its result and the verdict."
    write_table(NAME, body, caption, "tab:gate_variants_supp", env="table*", note=note, tabular="tabularx", width="\\linewidth", size="\\scriptsize\\renewcommand{\\arraystretch}{0.9}",
                colspec=">{\\raggedright\\arraybackslash}p{2.4cm} >{\\raggedright\\arraybackslash}p{1.1cm} >{\\raggedright\\arraybackslash}X "
                        ">{\\raggedright\\arraybackslash}X >{\\raggedright\\arraybackslash}X >{\\raggedright\\arraybackslash}p{1.9cm}")
    R.save()
    print(f"[{NAME}] 写好 {len(R.items)} 个数")


if __name__ == "__main__":
    main()
