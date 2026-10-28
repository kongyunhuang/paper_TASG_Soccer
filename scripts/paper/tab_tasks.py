#!/usr/bin/env python3
"""
tab_tasks.py, table of the eight tasks
======================================
Task definitions, positive labels, and the number of events and share of positives in the three data sets.
Training and validation (2022/23, 2023/24) counts and positive rates come from the [data] lines that build_dataset prints
into the training log. The 2024/25 counts of the 202 matches come from n_test in the single-task table of
results/u1_clean_summary.md, and their positive rates are recomputed from the per-event labels of the single-task models.
The 2025/26 counts of the 146 matches come from the last table of results/holdout_2526_summary.md. Task definitions
follow build_dataset in scripts/training/train_unified_clean.py.

Inputs   logs/u1_s0_training.log, results/u1_clean_summary.md, results/holdout_2526_summary.md,
         data/cache/single_clean/preds/*_test_meta.npz
Outputs  outputs/tables/tab_tasks.tex, outputs/provenance/tab_tasks_sources.json

Usage
  PYTHONPATH=. python -u scripts/paper/tab_tasks.py

Last modified 2026-10-28
"""
import re

import numpy as np

from scripts.paper.common import (DEV, SRC, TASK_NAME, TASKS, Registry, md_find, rel, tex, write_table)

NAME = "tab_tasks"

# positive labels, as defined in build_dataset of scripts/training/train_unified_clean.py (Tackle and Pressure through their own label functions)
DEFN = {
    "pass": "Pass completed (no outcome recorded)",
    "dest": "Cell of the pass end location on the $8 \\times 12$ grid (96 classes)",
    "shot": "Goal; penalties excluded",
    "interception": "Outcome Won or Success In Play",
    "ball_recovery": "Not flagged as a recovery failure",
    "pressure": "Pressing team performs a pass, carry, shot or dribble within 5\\,s in the same period",
    "dribble": "Outcome Complete",
    "tackle": "Tackle duel won (Won, Success, Success In Play, Success Out) against lost (Lost In Play, Lost Out)",
}
HOLD_TASKS = ["pass", "dest", "shot", "interception", "ball_recovery", "pressure"]


def main():
    R = Registry(NAME)
    log = SRC["train_log"].read_text(encoding="utf8").split("\n")
    pat = re.compile(r"\[data\] (\w+)\s+dev\s+([\d,]+) test\s+([\d,]+) mean label dev ([\d.]+) test ([\d.]+)")
    logrow = {}
    for i, line in enumerate(log):
        m = pat.search(line)
        if m and m.group(1) not in logrow:
            logrow[m.group(1)] = (i + 1, m)

    import os
    os.chdir(DEV)
    from scripts.eval.summarize_u1_clean import clean_matches
    clean = clean_matches()

    rows = []
    for r, t in enumerate(TASKS):
        cells = [TASK_NAME[t], DEFN[t]]
        ln, m = logrow[t]
        n_dev = int(m.group(2).replace(",", ""))
        R.add(f"{t}/train_n", n_dev, dict(kind="log", file=rel(SRC["train_log"]), line=ln, group=2), display=f"{n_dev:,}", cell=[r, 2], fmt="int")
        cells.append(f"{n_dev:,}")
        if t == "dest":
            cells.append("n/a")
        else:
            pr = float(m.group(4))
            R.add(f"{t}/train_pos", pr, dict(kind="log", file=rel(SRC["train_log"]), line=ln, group=4), display=f"{100 * pr:.1f}", cell=[r, 3], fmt="pct1")
            cells.append(f"{100 * pr:.1f}")
        # 202 matches of 2024/25
        ln2 = md_find(SRC["u1"], "## Single-task models on the same data (202", 0, task=t)
        n_test = R.md(f"{t}/test_n", SRC["u1"], ln2, "n_test", fmt="int", cell=[r, 4])
        cells.append(f"{int(n_test):,}")
        meta = np.load(DEV / "data" / "cache" / "single_clean" / "preds" / f"{t}_test_meta.npz", allow_pickle=True)
        ok = np.array([int(x) in clean for x in meta["match_id"]])
        assert int(ok.sum()) == int(n_test), f"{t} per-event count {ok.sum()} differs from n_test {n_test} of the summary table"
        if t == "dest":
            cells.append("n/a")
        else:
            pr2 = float(meta["y"][ok].astype(int).mean())
            R.perevent(f"{t}/test_pos", pr2, f"mean of y in data/cache/single_clean/preds/{t}_test_meta.npz over the 202 matches of clean_matches()", fmt="pct1", cell=[r, 5])
            cells.append(f"{100 * pr2:.1f}")
        # 146 matches of 2025/26
        if t in HOLD_TASKS:
            ln3 = md_find(SRC["hold_md"], "## Held-out events and label means", 0, **{"Task": t})
            n_h = R.md(f"{t}/hold_n", SRC["hold_md"], ln3, "Events", fmt="int", cell=[r, 6])
            cells.append(f"{int(n_h):,}")
            if t == "dest":
                cells.append("n/a")
            else:
                ph = R.md(f"{t}/hold_pos", SRC["hold_md"], ln3, "Label mean", fmt="pct1", cell=[r, 7])
                cells.append(f"{100 * ph:.1f}")
        else:
            cells += ["n.i.", "n.i."]
        rows.append(" & ".join(tex(c) if i >= 2 else c for i, c in enumerate(cells)) + " \\\\")

    body = ["\\toprule",
            " & & \\multicolumn{2}{c}{Train + val.} & \\multicolumn{2}{c}{Test 2024/25} & \\multicolumn{2}{c}{Held out 2025/26} \\\\",
            " & & \\multicolumn{2}{c}{2022/23, 2023/24} & \\multicolumn{2}{c}{202 matches} & \\multicolumn{2}{c}{146 matches} \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
            "Task & Positive label & Events & Pos.\\ (\\%) & Events & Pos.\\ (\\%) & Events & Pos.\\ (\\%) \\\\",
            "\\midrule"] + rows + ["\\bottomrule"]
    note = ("Events are StatsBomb events that carry a 360 freeze frame. Train + val.\\ (training and validation) are events from the 2022/23 and 2023/24 "
            "seasons of the Premier League and La Liga; each seed splits them by match into training and validation sets. For Pass, "
            "Destination, Ball recovery and Pressure they are a fixed random sample of 100,000 events. The 2024/25 test events of Pass, "
            "Destination and Pressure are a fixed random sample of 30,000 events drawn from 229 matches and then restricted to the 202 clean "
            "matches; the other tasks use all events of the 202 matches. The 2025/26 held-out set uses all events of its 146 matches "
            "(79 La Liga, 67 Premier League). Pos.\\ is the share of positive labels in percent. Destination is a 96-class label, so it has "
            "no positive rate (n/a). Dribble and Tackle were not included in the held-out set (n.i.) because their perspective-corrected "
            "tensors have not been built for 2025/26.")
    caption = "Task definitions, number of events and share of positive labels in the training, test and held-out data."
    write_table(NAME, body, caption, "tab:tasks", env="table*", note=note, tabular="tabularx", width="\\linewidth",
                colspec="l >{\\raggedright\\arraybackslash}X r r r r r r")
    R.save()
    print(f"[{NAME}] {len(R.items)} numbers written")


if __name__ == "__main__":
    main()
