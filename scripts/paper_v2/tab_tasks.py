#!/usr/bin/env python3
"""
表 tab_tasks  八个任务的定义、正例定义和三块数据各自的样本数与正例率
============================
训练加验证（22/23、23/24）的条数和正例率取训练日志里 build_dataset 打出的 [data] 行；
24/25 的 202 场取 u1_clean_summary.md 单任务表的 n_test，正例率从同数据单任务的逐事件标签重算；
25/26 的 146 场取 holdout_2526_summary.md 末尾“留出集条数和标签均值”一表。任务定义照 train_unified_clean.py 的 build_dataset。

输入  logs/u1_clean_u1_s0_2026-10-02.log，audit/u1_clean_summary.md，audit/holdout_2526_summary.md，data/cache/single_clean/preds/*_test_meta.npz
输出  outputs/tables/tab_tasks.tex，scripts/paper_v2/out/tab_tasks_sources.json

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/paper_v2/tab_tasks.py

Last modified 2026-10-06（表注措辞按 Miguel 10-05 修订）
"""
import re

import numpy as np

from scripts.paper_v2.common import (DEV, SRC, TASK_NAME, TASKS, Registry, md_find, rel, tex, write_table)

NAME = "tab_tasks"

# 正例定义，照 scripts/training/train_unified_clean.py 的 build_dataset（L183 到 L201，Tackle 见 L164 到 L166，Pressure 见 pressure_labels_5s）
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
        # 24/25 的 202 场
        ln2 = md_find(SRC["u1"], "## 同数据单任务模型（202", 0, task=t)
        n_test = R.md(f"{t}/test_n", SRC["u1"], ln2, "n_test", fmt="int", cell=[r, 4])
        cells.append(f"{int(n_test):,}")
        meta = np.load(DEV / "data" / "cache" / "single_clean" / "preds" / f"{t}_test_meta.npz", allow_pickle=True)
        ok = np.array([int(x) in clean for x in meta["match_id"]])
        assert int(ok.sum()) == int(n_test), f"{t} 逐事件条数 {ok.sum()} 和汇总表 n_test {n_test} 不一致"
        if t == "dest":
            cells.append("n/a")
        else:
            pr2 = float(meta["y"][ok].astype(int).mean())
            R.perevent(f"{t}/test_pos", pr2, f"data/cache/single_clean/preds/{t}_test_meta.npz 的 y，限 clean_matches() 的 202 场，取均值", fmt="pct1", cell=[r, 5])
            cells.append(f"{100 * pr2:.1f}")
        # 25/26 的 146 场
        if t in HOLD_TASKS:
            ln3 = md_find(SRC["hold_md"], "## 留出集条数和标签均值", 0, **{"任务": t})
            n_h = R.md(f"{t}/hold_n", SRC["hold_md"], ln3, "条数", fmt="int", cell=[r, 6])
            cells.append(f"{int(n_h):,}")
            if t == "dest":
                cells.append("n/a")
            else:
                ph = R.md(f"{t}/hold_pos", SRC["hold_md"], ln3, "标签均值", fmt="pct1", cell=[r, 7])
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
    print(f"[{NAME}] 写好 {len(R.items)} 个数")


if __name__ == "__main__":
    main()
