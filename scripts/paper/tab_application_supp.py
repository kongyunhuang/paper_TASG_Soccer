#!/usr/bin/env python3
"""
tab_application_supp.py, supplementary table of the player passing preference test
==================================================================================
D, its interval, the two folds, the three seeds and the verdict for all players and for players who changed team.
All numbers come from the table of the main quantity D in results/player_decision_preference_summary.md (rows for all
players and for players who changed team); the verdicts are also checked against results/decision_rules.md. D of the
players who stayed with their team goes into the table note (the summary marks it as not judged separately).

Inputs   results/player_decision_preference_summary.md, results/decision_rules.md
Outputs  outputs/tables/tab_application_supp.tex, outputs/provenance/tab_application_supp_sources.json

Usage
  PYTHONPATH=. python -u scripts/paper/tab_application_supp.py

Last modified 2026-10-28
"""
from scripts.paper.common import SRC, Registry, fmt_num, md_find, rule_line, tex, write_table

NAME = "tab_application_supp"
P = SRC["pref"]
NUM = r"([-+]?[0-9.]+)"


def main():
    R = Registry(NAME)
    head = "## Main quantity D"
    rows = []
    seed_txt = {}
    for r, (grp, label, verdict_en, verdict_cn) in enumerate([("All players", "All players", "Supported", "Supported"),
                                                               ("Changed team", "Changed team", "Inconclusive", "Inconclusive")]):
        ln = md_find(P, head, 0, **{"Group": grp})
        g = grp
        n = R.md(f"{g}/rows", P, ln, "Players (rows over both folds)", fmt="int", cell=[r, 1])
        d = R.md(f"{g}/D", P, ln, "D pooled", fmt="f5", cell=[r, 2])
        lo = R.mdre(f"{g}/D_lo", P, ln, "95% CI", r"^\[" + NUM, fmt="sf5", cell=[r, 3])
        hi = R.mdre(f"{g}/D_hi", P, ln, "95% CI", r", " + NUM + r"\]$", fmt="sf5", cell=[r, 3])
        f1 = R.mdre(f"{g}/fold1", P, ln, "First fold", r"^" + NUM, fmt="sf5", cell=[r, 4])
        f2 = R.mdre(f"{g}/fold2", P, ln, "Second fold", r"^" + NUM, fmt="sf5", cell=[r, 5])
        sd = [R.mdre(f"{g}/seed{k}", P, ln, "Pooled estimate per seed", r"^" + r"\S+\s+" * k + NUM, fmt="sf5", cell=["note", 0]) for k in range(3)]
        pct = R.mdre(f"{g}/pct", P, ln, "D share of global model log loss", r"^" + NUM + "%", fmt="f2", cell=[r, 6])
        R.text(f"{g}/verdict_summary", verdict_en, P, ln, verdict_cn, cell=[r, 7])
        crit_line = rule_line("preference_all_players") if grp == "All players" else rule_line("preference_changed_team")
        R.text(f"{g}/verdict_criteria_file", verdict_en, SRC["decision_rules"], crit_line, verdict_cn, cell=[r, 7])
        s5 = lambda v: tex(fmt_num(v, "sf5"))
        rows.append(" & ".join([label, f"{int(n):,}", tex(fmt_num(d, "f5")), f"[{s5(lo)}, {s5(hi)}]", s5(f1), s5(f2),
                                tex(fmt_num(pct, "f2")) + "\\%", verdict_en]) + " \\\\")
        seed_txt[label] = ", ".join(s5(x) for x in sd)
    # players who stayed (not judged separately) and the ratio of movers to stayers, for the table note
    ls = md_find(P, head, 0, **{"Group": "Stayed with team"})
    ds = R.md("Stayed with team/D", P, ls, "D pooled", fmt="f5", cell=["note", 0])
    L = P.read_text(encoding="utf8").split("\n")
    lr = next(i for i, x in enumerate(L) if x.startswith("D of players who changed team is")) + 1
    ratio = R.mdre("ratio_movers_stayers", P, lr, None, r"is ([0-9.]+) times", fmt="f2", cell=["note", 0])

    body = ["\\toprule",
            "Group & Rows & $D$ & 95\\% interval & Fold 1 & Fold 2 & Share & Verdict \\\\",
            "\\midrule"] + rows + ["\\bottomrule"]
    note = (f"$D$ is the per-pass log loss of a destination model with a player-specific preference term minus that of the same model "
            "with position and team-by-position terms only, so a negative $D$ means the player term adds predictive information. The terms "
            "act on five interpretable properties of each candidate cell (forward progress, lateral span, distance, opponents near the cell, "
            "long pass) on top of a frozen destination model with the resolution-preserving encoder. Fold 1 learns on 2022/23 and tests on "
            "2023/24; fold 2 learns on 2022/23 and 2023/24 and tests on the 202 clean matches of 2024/25. Rows counts players over the two "
            "folds, so a player who qualifies in both folds counts twice. The interval is a bootstrap over players (2000 resamples) for the pooled $D$, and the destination model was trained "
            f"with three seeds; the pooled estimates of seeds 0, 1 and 2 are {seed_txt['All players']} for all players and "
            f"{seed_txt['Changed team']} for players who changed team. Share is $D$ divided by the log loss of the destination model. A team change means the player "
            "passed most for different teams in the learning and test seasons; Changed team is the group of such players. The criterion written in advance counts $D$ as supported when "
            "the pooled interval lies below zero and both folds are negative, and lowers it to inconclusive if the pooled estimate of any "
            "seed is not negative. Players who stayed with their team have "
            f"$D = {fmt_num(ds, 'f5').replace('-', '-')}$ (not judged separately), and $D$ for players who changed team is {ratio:.2f} times "
            "that value.")
    caption = "Application, player-specific pass destination preference beyond team and position."
    write_table(NAME, body, caption, "tab:application_supp", env="table*", note=note, colspec="l r r c r r r l")
    R.save()
    print(f"[{NAME}] {len(R.items)} numbers written")


if __name__ == "__main__":
    main()
