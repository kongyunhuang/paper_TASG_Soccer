#!/usr/bin/env python3
"""
表 tab_interventions  打乱某任务的空间输入后重训，被干预任务的门控、名次、成绩和其余任务成绩的变化（逐种子）
============================
数全部取 u1_clean_summary.md 的“干预”一节（每行一个运行、种子、任务，列 gate_base、gate_shuffled_run、rank_base、
rank_shuffled_run、metric_base、metric_shuffled_run）。成绩变化是 metric_shuffled_run 减 metric_base，由表里的四位小数相减；
落点干预的三行另和“落点干预下其余任务被拖累多少”一节的门控行逐格核对。表注里门控和拼接的其余七任务平均变化取同一节。

输入  audit/u1_clean_summary.md
输出  outputs/tables/tab_interventions.tex，scripts/paper_v2/out/tab_interventions_sources.json

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/paper_v2/tab_interventions.py

Last modified 2026-10-06（表注措辞按合作者 10-05 修订）
"""
from scripts.paper_v2.common import (SRC, TASK_NAME, TASKS, Registry, fmt_num, md_find, md_value, tex, unified_seed_values,
                                     write_table)

NAME = "tab_interventions"
RUNS = [("shuf_dest", "dest", [0, 1, 2]), ("shuf_tackle", "tackle", [0, 1, 2]), ("shuf_pressure", "pressure", [0])]
HEAD = "## 干预"
HEAD_DRAG = "## 落点干预下其余任务被拖累多少"


def main():
    R = Registry(NAME)
    U = SRC["u1"]
    rows = []
    r = 0
    for run, task, seeds in RUNS:
        for sd in seeds:
            ln = md_find(U, HEAD, 0, run=run, seed=sd, task=task)
            p = f"{run}/s{sd}"
            gb = R.md(p + "/gate_base", U, ln, "gate_base", fmt="f4", cell=[r, 2])
            gs = R.md(p + "/gate_after", U, ln, "gate_shuffled_run", fmt="f4", cell=[r, 2])
            rb = R.md(p + "/rank_base", U, ln, "rank_base", fmt="int", cell=[r, 3])
            rs = R.md(p + "/rank_after", U, ln, "rank_shuffled_run", fmt="int", cell=[r, 3])
            mb = R.md(p + "/metric_base", U, ln, "metric_base", fmt="f4", cell=[r, 4])
            ms = R.md(p + "/metric_after", U, ln, "metric_shuffled_run", fmt="f4", cell=[r, 4])
            raw_s, raw_b = unified_seed_values(run, task, "metric")[sd], unified_seed_values("u1", task, "metric")[sd]
            assert round(raw_s, 4) == round(ms, 4) and round(raw_b, 4) == round(mb, 4), f"{p} 逐事件重算和干预表不一致"
            dm = raw_s - raw_b
            R.perevent(p + "/metric_change", dm, f"{run}_s{sd}.npz 减 u1_s{sd}.npz 的 {task} 指标，202 场，用未四舍五入的值相减"
                       f"（两者四舍五入后分别等于干预表 L{ln} 的 metric_shuffled_run 和 metric_base）", fmt="f4", cell=[r, 5])
            # 其余 AUC 任务的变化，取绝对值最大的一个
            others = []
            for t in TASKS:
                if t in (task, "dest"):
                    continue
                lo = md_find(U, HEAD, 0, run=run, seed=sd, task=t)
                a, b = unified_seed_values(run, t, "metric")[sd], unified_seed_values("u1", t, "metric")[sd]
                assert round(a, 4) == round(md_value(U, lo, "metric_shuffled_run"), 4) and round(b, 4) == round(md_value(U, lo, "metric_base"), 4)
                others.append((a - b, t, lo))
            big = max(others, key=lambda x: (abs(x[0]), x[1]))
            R.perevent(f"{p}/others_max_change", big[0], f"{big[1]} 上 {run}_s{sd} 减 u1_s{sd} 的 AUC（202 场，未四舍五入），"
                       f"是其余 {len(others)} 个 AUC 任务里绝对值最大的；各任务两端的值见干预表 L{min(x[2] for x in others)} 到 L{max(x[2] for x in others)}",
                       fmt="f4", cell=[r, 6])
            if task == "dest":
                ddest = "n/a"
                # 和“拖累”一节门控行逐格核对
                lg = md_find(U, HEAD_DRAG, 0, seed=sd, fusion="门控")
                for dv, t, _ in others + [(dm, "dest", None)]:
                    v = md_value(U, lg, t)
                    assert round(dv, 4) == round(v, 4), f"{p} {t} 逐事件相减 {dv:.4f} 和拖累表 {v} 不一致"
                R.notes.append(f"{p} 七个其余任务和落点的变化与拖累表 L{lg} 逐格一致")
            else:
                lo = md_find(U, HEAD, 0, run=run, seed=sd, task="dest")
                a, b = unified_seed_values(run, "dest", "metric")[sd], unified_seed_values("u1", "dest", "metric")[sd]
                assert round(a, 4) == round(md_value(U, lo, "metric_shuffled_run"), 4) and round(b, 4) == round(md_value(U, lo, "metric_base"), 4)
                dd = a - b
                R.perevent(f"{p}/dest_change", dd, f"dest 上 {run}_s{sd} 减 u1_s{sd} 的 top-1（202 场，未四舍五入），两端见干预表 L{lo}", fmt="f4", cell=[r, 7])
                ddest = tex(fmt_num(dd, "f4"))
            label = TASK_NAME[task] if sd == seeds[0] else ""
            rows.append(" & ".join([label, str(sd), f"{gb:.4f} $\\to$ {gs:.4f}", f"{int(rb)} $\\to$ {int(rs)}",
                                    f"{mb:.4f} $\\to$ {ms:.4f}", tex(fmt_num(dm, "f4")),
                                    f"{tex(fmt_num(big[0], 'f4'))} ({TASK_NAME[big[1]]})", ddest]) + " \\\\")
            r += 1
        rows.append("\\addlinespace")
    rows = rows[:-1]

    # 表注里的门控和拼接，其余七任务平均变化（落点干预）
    avg = {}
    for fus in ["门控", "拼接"]:
        for sd in [0, 1, 2]:
            lg = md_find(U, HEAD_DRAG, 0, seed=sd, fusion=fus)
            avg[(fus, sd)] = R.md(f"drag/{fus}/s{sd}", U, lg, "other7_avg_change", fmt="f4", cell=["note", 0])
    g = ", ".join(tex(fmt_num(avg[("门控", s)], "f4")) for s in [0, 1, 2])
    c = ", ".join(tex(fmt_num(avg[("拼接", s)], "f4")) for s in [0, 1, 2])

    body = ["\\toprule",
            "Shuffled task & Seed & Gate & Rank & Metric & Change & \\makecell{Largest change,\\\\other outcome tasks} & \\makecell{Destination\\\\top-1 change} \\\\",
            "\\midrule"] + rows + ["\\bottomrule"]
    note = ("Each intervention retrains the unified model (pooled encoder, thin scalars) after shuffling the spatial tensors of one task "
            "among that task's events, so the tensors carry no information about that task; the other tasks are left unchanged. Each "
            "entry reads base $\\to$ shuffled, where base is the unshuffled unified model with the same seed, and all values are computed "
            "on the 202 test matches of 2024/25. Gate is the gate value of the shuffled task, that is, the weight on the spatial encoding "
            "averaged over the 64 fused dimensions and the task's test events. Rank orders the gate among the eight tasks (1 is the "
            "highest). The metric is top-1 for Destination and AUC otherwise; Change is shuffled minus base. The largest change among the other "
            "outcome tasks is the AUC change with the largest absolute value; the destination top-1 change is not applicable (n/a) in the "
            "Destination rows, where it is the change of the shuffled task itself. Changes are computed before rounding. The Pressure intervention was run with one seed. Under the "
            f"destination intervention the other seven tasks change on average by {g} (seeds 0, 1, 2) with the gate and by {c} when the "
            "gate is replaced by concatenation, so their stability is not specific to the gate.")
    caption = ("Gate value, gate rank and metric of the intervened task before and after shuffling its spatial input, per seed, "
               "and the change on the other tasks.")
    write_table(NAME, body, caption, "tab:interventions", env="table*", note=note, colspec="l c c c c c c c")
    R.save()
    print(f"[{NAME}] 写好 {len(R.items)} 个数")


if __name__ == "__main__":
    main()
