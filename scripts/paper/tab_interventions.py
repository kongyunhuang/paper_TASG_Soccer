#!/usr/bin/env python3
"""
tab_interventions.py, table of the spatial input interventions
==============================================================
Per seed, the change of gate, gate rank and metric of the intervened task after its spatial input is shuffled and the
model retrained, and the change on the other tasks.
All numbers come from the intervention section of results/u1_clean_summary.md (one row per run, seed and task, columns
gate_base, gate_shuffled_run, rank_base, rank_shuffled_run, metric_base, metric_shuffled_run). The metric change is
metric_shuffled_run minus metric_base, recomputed from the per-event files before rounding. The three rows of the
destination intervention are also checked cell by cell against the gate rows of the section on the loss of the other
tasks under the destination intervention, and the mean change of the other seven tasks with the gate and with
concatenation in the table note comes from that section.

Inputs   results/u1_clean_summary.md, data/cache/u1_clean/*.npz
Outputs  outputs/tables/tab_interventions.tex, outputs/provenance/tab_interventions_sources.json

Usage
  PYTHONPATH=. python -u scripts/paper/tab_interventions.py

Last modified 2026-10-28
"""
from scripts.paper.common import (SRC, TASK_NAME, TASKS, Registry, fmt_num, md_find, md_value, tex, unified_seed_values,
                                  write_table)

NAME = "tab_interventions"
RUNS = [("shuf_dest", "dest", [0, 1, 2]), ("shuf_tackle", "tackle", [0, 1, 2]), ("shuf_pressure", "pressure", [0])]
HEAD = "## Interventions"
HEAD_DRAG = "## Loss of the other tasks under the destination intervention"


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
            assert round(raw_s, 4) == round(ms, 4) and round(raw_b, 4) == round(mb, 4), f"{p} per-event recomputation differs from the intervention table"
            dm = raw_s - raw_b
            R.perevent(p + "/metric_change", dm, f"{run}_s{sd}.npz minus u1_s{sd}.npz, {task} metric, 202 matches, difference of unrounded values "
                       f"(after rounding they equal metric_shuffled_run and metric_base in L{ln} of the intervention table)", fmt="f4", cell=[r, 5])
            # change on the other AUC tasks, the one with the largest absolute value
            others = []
            for t in TASKS:
                if t in (task, "dest"):
                    continue
                lo = md_find(U, HEAD, 0, run=run, seed=sd, task=t)
                a, b = unified_seed_values(run, t, "metric")[sd], unified_seed_values("u1", t, "metric")[sd]
                assert round(a, 4) == round(md_value(U, lo, "metric_shuffled_run"), 4) and round(b, 4) == round(md_value(U, lo, "metric_base"), 4)
                others.append((a - b, t, lo))
            big = max(others, key=lambda x: (abs(x[0]), x[1]))
            R.perevent(f"{p}/others_max_change", big[0], f"{big[1]}, AUC of {run}_s{sd} minus u1_s{sd} (202 matches, unrounded), "
                       f"the largest absolute change among the other {len(others)} AUC tasks; the values of each task are in L{min(x[2] for x in others)} to L{max(x[2] for x in others)} of the intervention table",
                       fmt="f4", cell=[r, 6])
            if task == "dest":
                ddest = "n/a"
                # checked cell by cell against the gate row of the section on the loss of the other tasks
                lg = md_find(U, HEAD_DRAG, 0, seed=sd, fusion="gate")
                for dv, t, _ in others + [(dm, "dest", None)]:
                    v = md_value(U, lg, t)
                    assert round(dv, 4) == round(v, 4), f"{p} {t} per-event difference {dv:.4f} differs from the table of the other tasks {v}"
                R.notes.append(f"{p} changes of the seven other tasks and of the destination equal L{lg} of the table of the other tasks cell by cell")
            else:
                lo = md_find(U, HEAD, 0, run=run, seed=sd, task="dest")
                a, b = unified_seed_values(run, "dest", "metric")[sd], unified_seed_values("u1", "dest", "metric")[sd]
                assert round(a, 4) == round(md_value(U, lo, "metric_shuffled_run"), 4) and round(b, 4) == round(md_value(U, lo, "metric_base"), 4)
                dd = a - b
                R.perevent(f"{p}/dest_change", dd, f"top-1 of {run}_s{sd} minus u1_s{sd} on dest (202 matches, unrounded), both values in L{lo} of the intervention table", fmt="f4", cell=[r, 7])
                ddest = tex(fmt_num(dd, "f4"))
            label = TASK_NAME[task] if sd == seeds[0] else ""
            rows.append(" & ".join([label, str(sd), f"{gb:.4f} $\\to$ {gs:.4f}", f"{int(rb)} $\\to$ {int(rs)}",
                                    f"{mb:.4f} $\\to$ {ms:.4f}", tex(fmt_num(dm, "f4")),
                                    f"{tex(fmt_num(big[0], 'f4'))} ({TASK_NAME[big[1]]})", ddest]) + " \\\\")
            r += 1
        rows.append("\\addlinespace")
    rows = rows[:-1]

    # mean change of the other seven tasks under the destination intervention, with the gate and with concatenation, for the table note
    avg = {}
    for fus in ["gate", "concat"]:
        for sd in [0, 1, 2]:
            lg = md_find(U, HEAD_DRAG, 0, seed=sd, fusion=fus)
            avg[(fus, sd)] = R.md(f"drag/{fus}/s{sd}", U, lg, "other7_avg_change", fmt="f4", cell=["note", 0])
    g = ", ".join(tex(fmt_num(avg[("gate", s)], "f4")) for s in [0, 1, 2])
    c = ", ".join(tex(fmt_num(avg[("concat", s)], "f4")) for s in [0, 1, 2])

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
    print(f"[{NAME}] {len(R.items)} numbers written")


if __name__ == "__main__":
    main()
