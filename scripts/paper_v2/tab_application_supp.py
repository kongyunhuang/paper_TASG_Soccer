#!/usr/bin/env python3
"""
补充材料表 tab_application_supp  球员传球决策偏好的正式检验，全体球员和换队球员两组的 D、区间、两折、三个种子和判定
============================
数全部取 audit/player_decision_preference_summary.md 的“主要的量 D”一表（全体球员、换队球员两行），
判定词和标准取 memory_logs/2026-10-02_球员决策偏好_预先写下的判定标准.md，留队球员的 D 进表注（汇总表写明不单独判）。

输入  audit/player_decision_preference_summary.md，球员决策偏好标准文件
输出  outputs/tables/tab_application_supp.tex，scripts/paper_v2/out/tab_application_supp_sources.json

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/paper_v2/tab_application_supp.py

Last modified 2026-10-04
"""
from scripts.paper_v2.common import SRC, Registry, fmt_num, md_find, tex, write_table

NAME = "tab_application_supp"
P = SRC["pref"]
NUM = r"([-+]?[0-9.]+)"


def main():
    R = Registry(NAME)
    head = "## 主要的量 D"
    rows = []
    seed_txt = {}
    for r, (grp, label, verdict_en, verdict_cn) in enumerate([("全体球员", "All players", "Supported", "成立"),
                                                               ("换队球员", "Changed team", "Inconclusive", "不确定")]):
        ln = md_find(P, head, 0, **{"组": grp})
        g = grp
        n = R.md(f"{g}/rows", P, ln, "球员数（两折合计行数）", fmt="int", cell=[r, 1])
        d = R.md(f"{g}/D", P, ln, "D 合并", fmt="f5", cell=[r, 2])
        lo = R.mdre(f"{g}/D_lo", P, ln, "95% 区间", r"^\[" + NUM, fmt="sf5", cell=[r, 3])
        hi = R.mdre(f"{g}/D_hi", P, ln, "95% 区间", r", " + NUM + r"\]$", fmt="sf5", cell=[r, 3])
        f1 = R.mdre(f"{g}/fold1", P, ln, "折一", r"^" + NUM, fmt="sf5", cell=[r, 4])
        f2 = R.mdre(f"{g}/fold2", P, ln, "折二", r"^" + NUM, fmt="sf5", cell=[r, 5])
        sd = [R.mdre(f"{g}/seed{k}", P, ln, "各种子合并点估计", r"^" + r"\S+\s+" * k + NUM, fmt="sf5", cell=["note", 0]) for k in range(3)]
        pct = R.mdre(f"{g}/pct", P, ln, "D 占全局模型对数损失", r"^" + NUM + "%", fmt="f2", cell=[r, 6])
        R.text(f"{g}/verdict_summary", verdict_en, P, ln, verdict_cn, cell=[r, 7])
        crit_line = 80 if grp == "全体球员" else 81
        R.text(f"{g}/verdict_criteria_file", verdict_en, SRC["crit_pref"], crit_line, verdict_cn, cell=[r, 7])
        s5 = lambda v: tex(fmt_num(v, "sf5"))
        rows.append(" & ".join([label, f"{int(n):,}", tex(fmt_num(d, "f5")), f"[{s5(lo)}, {s5(hi)}]", s5(f1), s5(f2),
                                tex(fmt_num(pct, "f2")) + "\\%", verdict_en]) + " \\\\")
        seed_txt[label] = ", ".join(s5(x) for x in sd)
    # 留队球员（不单独判）和换队对留队的比值，进表注
    ls = md_find(P, head, 0, **{"组": "留队球员"})
    ds = R.md("留队球员/D", P, ls, "D 合并", fmt="f5", cell=["note", 0])
    L = P.read_text(encoding="utf8").split("\n")
    lr = next(i for i, x in enumerate(L) if x.startswith("换队球员的 D 是留队球员的")) + 1
    ratio = R.mdre("ratio_movers_stayers", P, lr, None, r"留队球员的 ([0-9.]+) 倍", fmt="f2", cell=["note", 0])

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
    print(f"[{NAME}] 写好 {len(R.items)} 个数")


if __name__ == "__main__":
    main()
