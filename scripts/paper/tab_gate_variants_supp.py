#!/usr/bin/env python3
"""
tab_gate_variants_supp.py, supplementary table of the gate variants and controls
================================================================================
One row per experiment with a rule fixed in advance, giving the setting, the question, the condition to be supported,
the result and the verdict.
Numbers come from the matching sections of results/u1_clean_summary.md (ablation control, enriched scalars against the
baseline, confirmation step of the gate penalty, selection step of the hard gate, concatenation control,
interventions). The three-seed numbers of the accuracy question of the enriched scalars, the destination top-1 of the
encoder rule and every verdict come from results/decision_rules.md.

Inputs   results/u1_clean_summary.md, results/decision_rules.md
Outputs  outputs/tables/tab_gate_variants_supp.tex, outputs/provenance/tab_gate_variants_supp_sources.json

Usage
  PYTHONPATH=. python -u scripts/paper/tab_gate_variants_supp.py

Last modified 2026-10-28
"""
from scripts.paper.common import SRC, Registry, fmt_num, md_find, md_value, rule_line, tex, write_table

NAME = "tab_gate_variants_supp"
U = SRC["u1"]


def main():
    R = Registry(NAME)
    rows = []

    def lst(vals, fmt):
        return ", ".join(tex(fmt_num(v, fmt)) for v in vals)

    # 1 two-channel spatial input
    r = 0
    sp, pr = [], []
    for sd in [0, 1, 2]:
        ln = md_find(U, "## Gate mean against the loss from shuffling the player channels", 0, run="u1_2ch", seed=sd)
        sp.append(R.md(f"2ch/s{sd}/spearman", U, ln, "spearman_gate_vs_player_drop", fmt="f3", cell=[r, 4]))
        pr.append(R.md(f"2ch/s{sd}/pressure_rank", U, ln, "pressure_gate_rank", fmt="int", cell=[r, 4]))
    v1 = R.text("2ch/verdict", "Inconclusive", SRC["decision_rules"], rule_line("two_channel"), "Inconclusive", cell=[r, 5])
    rows.append(["Two-channel spatial input (players only)", "0, 1, 2",
                 "Does the gate follow how much each task loses when player positions are shuffled?",
                 "Spearman between gate and that loss $\\geq$ 0.70 and Pressure gate in the top four, in every seed",
                 f"Spearman {lst(sp, 'f3')}; Pressure gate rank {', '.join(str(int(x)) for x in pr)}", v1])

    # 2 enriched scalars, gate question
    r = 1
    dr, td, dc = [], [], []
    for sd in [0, 1, 2]:
        ln = md_find(U, "## Unified model with enriched scalars (u1rich) against the baseline", 0, seed=sd)
        dr.append(R.md(f"rich_gate/s{sd}/drop", U, ln, "outcome_gate_avg_drop", fmt="f3", cell=[r, 4]))
        td.append(R.mdre(f"rich_gate/s{sd}/tasks_down", U, ln, "tasks_down", r"^(\d+)/7", fmt="int", cell=[r, 4]))
        dc.append(R.md(f"rich_gate/s{sd}/dest_change", U, ln, "dest_gate_change", fmt="f3", cell=[r, 4]))
    v2 = R.text("rich_gate/verdict", "Inconclusive; the expected effect was not seen", SRC["decision_rules"], rule_line("enriched_gate"), "Inconclusive; the expected effect was not seen", cell=[r, 5])
    rows.append(["Enriched scalars in the unified model, gate", "0, 1, 2",
                 "With enriched scalars, do the outcome-task gates close while the destination gate stays?",
                 "Mean outcome gate falls by $\\geq$ 0.10, at least 5 of 7 outcome gates fall, destination gate changes by $<$ 0.10 and stays first, in every seed",
                 f"Fall {lst(dr, 'f3')}; outcome gates falling {', '.join(str(int(x)) for x in td)} of 7; destination gate change {lst(dc, 'f3')}", v2])

    # 3 enriched scalars, accuracy question
    r = 2
    lp = rule_line("Pass")
    ls = rule_line("Shot")
    p3 = R.md("rich_perf/3seeds/pass", SRC["decision_rules"], lp, "enriched unified minus enriched XGBoost", fmt="f4", cell=[r, 4])
    s3 = R.md("rich_perf/3seeds/shot", SRC["decision_rules"], ls, "enriched unified minus enriched XGBoost", fmt="f4", cell=[r, 4])
    l4p = md_find(U, "## Unified model with enriched scalars (u1rich) against the baseline", 1, task="pass")
    l4s = md_find(U, "## Unified model with enriched scalars (u1rich) against the baseline", 1, task="shot")
    p4 = R.md("rich_perf/4seeds/pass", U, l4p, "rich_minus_rich_xgb", fmt="f4", cell=[r, 4])
    s4 = R.md("rich_perf/4seeds/shot", U, l4s, "rich_minus_rich_xgb", fmt="f4", cell=[r, 4])
    nok = R.mdre("rich_perf/n_ok", U, md_find(U, "## Unified model with enriched scalars (u1rich) against the baseline", 1, task="tackle") + 4, None,
           r"only (\d)/7 tasks with a difference of at least -0\.003", fmt="int", cell=[r, 4])
    v3 = R.text("rich_perf/verdict", "Not matched", SRC["decision_rules"], rule_line("enriched_accuracy"), "Not matched", cell=[r, 5])
    rows.append(["Enriched scalars in the unified model, AUC", "0, 1, 2 (also 3)",
                 "Does the enriched unified model match enriched XGBoost on the outcome tasks?",
                 "At least 6 of 7 outcome tasks no more than 0.003 below enriched XGBoost",
                 f"{int(nok)} of 7; Pass {tex(fmt_num(p3, 'f4'))}, Shot {tex(fmt_num(s3, 'f4'))} with three seeds; Pass {tex(fmt_num(p4, 'f4'))}, "
                 f"Shot {tex(fmt_num(s4, 'f4'))} and still 5 of 7 with four seeds", v3])

    # 4 soft gate penalty, enriched scalars
    r = 3
    og, tw, da = [], [], []
    for sd in [1, 2, 3]:
        ln = md_find(U, "### Soft gate, enriched scalars, confirmation step", 0, seed=sd)
        og.append(R.md(f"pen_rich/s{sd}/outcome_gate", U, ln, "outcome_gate_avg", fmt="f3", cell=[r, 4]))
        tw.append(R.mdre(f"pen_rich/s{sd}/within", U, ln, "tasks_within_tolerance", r"^(\d)/8", fmt="int", cell=[r, 4]))
        da.append(R.md(f"pen_rich/s{sd}/drop_all", U, ln, "drop_all_outcome_avg", fmt="f3", cell=[r, 4]))
    v4 = R.text("pen_rich/verdict", "Inconclusive", SRC["decision_rules"], rule_line("soft_penalty_enriched"), "Inconclusive", cell=[r, 5])
    rows.append(["Soft gate with an opening penalty ($\\lambda = 0.01$), enriched scalars", "1, 2, 3",
                 "With a price on opening and enriched scalars, do the outcome gates close without loss of AUC?",
                 "Mean outcome gate $\\leq$ 0.10 with the destination gate $\\geq$ 0.50 and first; at least 7 of 8 tasks within tolerance; "
                 "AUC loss of the outcome tasks under a full tensor shuffle $\\leq$ 0.01; in every seed",
                 f"Mean outcome gate {lst(og, 'f3')}; tasks within tolerance {', '.join(str(int(x)) for x in tw)} of 8; loss under shuffle {lst(da, 'f3')}", v4])

    # 5 soft gate penalty, thin scalars
    r = 4
    s8, s7, tw2 = [], [], []
    for sd in [1, 2, 3]:
        ln = md_find(U, "### Soft gate, thin scalars, confirmation step", 0, seed=sd)
        s8.append(R.md(f"pen_thin/s{sd}/spearman8", U, ln, "spearman_8", fmt="f2", cell=[r, 4]))
        s7.append(R.md(f"pen_thin/s{sd}/spearman7", U, ln, "spearman_7", fmt="f2", cell=[r, 4]))
        tw2.append(R.mdre(f"pen_thin/s{sd}/within", U, ln, "tasks_within_tolerance", r"^(\d)/8", fmt="int", cell=[r, 4]))
    v5 = R.text("pen_thin/verdict", "Not supported", SRC["decision_rules"], rule_line("soft_penalty_thin"), "Not supported", cell=[r, 5])
    rows.append(["Soft gate with an opening penalty ($\\lambda = 0.01$), thin scalars", "1, 2, 3",
                 "With a price on opening, does the gate order the tasks by their single-task CNN gain?",
                 "Spearman with that gain $\\geq$ 0.70 over eight tasks and $\\geq$ 0.60 over the seven AUC tasks; at least 7 of 8 tasks within tolerance; in every seed",
                 f"Spearman {lst(s8, 'f2')} (eight tasks) and {lst(s7, 'f2')} (seven); tasks within tolerance {', '.join(str(int(x)) for x in tw2)} of 8", v5])

    # 6 and 7 hard gate
    for r, (key, head, crit_line, crit_txt, lab) in enumerate([
            ("hard_rich", "### Hard gate, enriched scalars, selection step", rule_line("hard_gate_enriched"), "Not established; stopped at the lambda selection step", "enriched scalars"),
            ("hard_thin", "### Hard gate, thin scalars, selection step", rule_line("hard_gate_thin"), "Not established; stopped at the lambda selection step", "thin scalars")], start=5):
        vb = R.md(f"{key}/val_base", U, md_find(U, head, 0, lam="0"), "val_score", fmt="f4", cell=[r, 4])
        v01 = R.md(f"{key}/val_001", U, md_find(U, head, 0, lam="0.01"), "val_score", fmt="f4", cell=[r, 4])
        v03 = R.md(f"{key}/val_003", U, md_find(U, head, 0, lam="0.03"), "val_score", fmt="f4", cell=[r, 4])
        vv = R.text(f"{key}/verdict", "Not established; stopped at the $\\lambda$ selection step", SRC["decision_rules"], crit_line, crit_txt, cell=[r, 5])
        rows.append([f"Hard gate with an opening penalty, {lab}", "0 (selection)",
                     "Does a binary gate, which cannot pass a small weight on, give a gate reading with meaning?",
                     "A $\\lambda$ enters the confirmation step only if its validation score is within 0.002 of the model without penalty",
                     f"Validation score {v01:.4f} ($\\lambda = 0.01$) and {v03:.4f} ($\\lambda = 0.03$) against {vb:.4f}; no $\\lambda$ kept", vv])

    # 8 concatenation control
    r = 7
    ex = []
    for sd in [0, 1, 2]:
        lg = md_find(U, "## Loss of the other tasks under the destination intervention", 0, seed=sd, fusion="gate")
        lc = md_find(U, "## Loss of the other tasks under the destination intervention", 0, seed=sd, fusion="concat")
        g, c = md_value(U, lg, "other7_avg_change"), md_value(U, lc, "other7_avg_change")
        ex.append(R.derived(f"concat/s{sd}/extra_loss", round(g - c, 4), f"other7_avg_change of the gate row L{lg} minus that of the concatenation row L{lc}",
                            [f"md:{lg}", f"md:{lc}"], fmt="f4", cell=[r, 4]))
    R.mdre("concat/summary_line", U, md_find(U, "## Loss of the other tasks under the destination intervention", 0, seed=2, fusion="concat") + 4, None,
           r"\(per seed\) \[([-0-9.]+),", fmt="f4", cell=None)
    v8 = R.text("concat/verdict", "Not supported", SRC["decision_rules"], rule_line("concat_control"), "Not supported", cell=[r, 5])
    rows.append(["Concatenation in place of the gate, destination shuffled", "0, 1, 2",
                 "Under the destination intervention, does the gate protect the other tasks better than concatenation?",
                 "With concatenation the other seven tasks lose at least 0.005 more AUC on average than with the gate, in every seed",
                 f"Extra loss with concatenation {lst(ex, 'f4')}", v8])

    # 9 resolution-preserving encoder
    r = 8
    li = md_find(U, "## Interventions", 0, run="u1fcn_shuf_dest", seed=0, task="dest")
    gch = R.md("fcn/gate_change", U, li, "gate_change", fmt="f3", cell=[r, 4])
    rb = R.md("fcn/rank_base", U, li, "rank_base", fmt="int", cell=[r, 4])
    ra = R.md("fcn/rank_after", U, li, "rank_shuffled_run", fmt="int", cell=[r, 4])
    t1 = [R.md(f"fcn/top1_s{k}", SRC["decision_rules"], rule_line(f"main_encoder/top1_seed{k}"), "value", fmt="f1", cell=[r, 4])
          for k in range(3)]
    v9 = R.text("fcn/verdict", "Not met; the pooled encoder is the main model", SRC["decision_rules"], rule_line("main_encoder"), "Not met; the pooled encoder is the main model", cell=[r, 5])
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
    print(f"[{NAME}] {len(R.items)} numbers written")


if __name__ == "__main__":
    main()
