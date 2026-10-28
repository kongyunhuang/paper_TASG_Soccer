#!/usr/bin/env python3
"""
tab_holdout.py, table of the held-out confirmation on 2025/26
=============================================================
For each of the four claims on the 146 held-out matches, the quantities, their 2024/25 and 2025/26 values, the
threshold to confirm and the verdict.
Quantities and intervals come from the sections C1 to C4 of results/holdout_2526_summary.md (registered by line
number), thresholds from the threshold dictionary at the top of the same file (equal to the rules of the summary json),
and verdicts from the recorded verdicts of results/decision_rules.md, without a summary sentence.
The 2024/25 gate minimum and maximum of C4 are the per-seed values of the u1_clean per-event files rounded to three
decimals (checked to four decimals against gate_min and gate_max of the tables of runs u1 and shuf_dest in
results/u1_clean_summary.md), and the ranks come from those tables; the 2025/26 values come from derived.C4 of the
summary json. Every 2025/26 number is also checked against the derived fields of the summary json to four decimals.

Inputs   results/holdout_2526_summary.md, results/holdout_2526_summary.json, results/u1_clean_summary.md,
         results/decision_rules.md, data/cache/u1_clean/{u1,shuf_dest}_s*.npz
Outputs  outputs/tables/tab_holdout.tex, outputs/provenance/tab_holdout_sources.json

Usage
  PYTHONPATH=. python -u scripts/paper/tab_holdout.py

Last modified 2026-10-28
"""
import re

from scripts.paper.common import (SRC, Registry, fmt_num, json_value, md_find, md_row, md_value, rel, rule_line, tex, unified_seed_values,
                                  write_table)

NAME = "tab_holdout"
H, J, U, F = SRC["hold_md"], SRC["hold_json"], SRC["u1"], SRC["decision_rules"]


def lines(p):
    return p.read_text(encoding="utf8").split("\n")


def row_line(path, head, label):
    """Line number of the row whose first column equals label, in the table of section head"""
    return md_find(path, head, 0, **{"Quantity": label})


def main():
    R = Registry(NAME)
    rows = []
    r = [0]

    def emit(claim_cells):
        rows.append(" & ".join(claim_cells) + " \\\\")
        r[0] += 1

    def ci_txt(lo, hi):
        return f"[{tex(fmt_num(lo, 'f4'))}, {tex(fmt_num(hi, 'f4'))}]"

    def check(v, jpath):
        jv = json_value(J, jpath)
        assert round(jv, 4) == round(v, 4), f"{jpath} json {jv} differs from md {v}"

    # verdicts, from the recorded verdicts of the decision rules
    verdict_line = {c: rule_line(f"holdout_{c}") for c in ["C1", "C2", "C3", "C4"]}
    VERD = {"C1": ("Confirmed", "Confirmed"), "C2": ("Confirmed, Pressure near the bound", "Confirmed, Pressure near the bound"),
            "C3": ("Inconclusive", "Inconclusive"), "C4": ("Holds", "Holds")}

    def verdict(c, rr):
        txt, must = VERD[c]
        R.text(f"{c}/verdict", txt, F, verdict_line[c], must, cell=[rr, 5])
        if c == "C2":
            R.text("C2/verdict_nearline", "near the bound", F, verdict_line[c], "against the bound of 0.010", cell=[rr, 5])
        return txt

    # line of a threshold in the threshold dictionary (code block at the top of the held-out report)
    HL = lines(H)

    def thr_line(key):
        return next(i for i, x in enumerate(HL) if f'"{key}"' in x) + 1

    def thr(iid, key, cpath, rr):
        ln = thr_line(key)
        v = float(re.search(r":\s*([-0-9.]+)", HL[ln - 1]).group(1))
        assert v == json_value(J, ["rules"] + cpath), f"{key} threshold differs from the json"
        R.add(iid, v, dict(kind="mdline", file=rel(H), line=ln, regex=r":\s*([-0-9.]+)"), display=fmt_num(v, "f3"), cell=[rr, 4], fmt="f3")
        return v

    # ---------------- C1
    rows.append("\\multicolumn{6}{l}{\\textit{C1. The spatial structure cannot be replaced for pass destination}} \\\\")
    head = "## C1"
    for label, iid, thrkey, jkey, text in [
        ("u1fcn minus strongest rule", "C1/fcn_minus_rule", "fcn_minus_rule_confirm", "fcn_minus_rule", "Variant minus strongest rule, top-1"),
        ("u1fcn minus M2 enriched", "C1/fcn_minus_m2rich", "fcn_minus_m2rich_confirm", "fcn_minus_m2rich", "Variant minus enriched-scalar MLP, top-1")]:
        ln = row_line(H, head, label)
        rr = r[0]
        a = R.md(iid + "/2425", H, ln, "24/25", fmt="f4", cell=[rr, 2])
        b = R.md(iid + "/2526", H, ln, "Held-out", fmt="f4", cell=[rr, 3])
        lo = R.md(iid + "/2526_lo", H, ln, "95% CI", "lo", fmt="f4", cell=[rr, 3])
        hi = R.md(iid + "/2526_hi", H, ln, "95% CI", "hi", fmt="f4", cell=[rr, 3])
        check(b, ["derived", "C1", jkey])
        check(lo, ["ci", "derived", "C1", jkey, 0]); check(hi, ["ci", "derived", "C1", jkey, 1])
        t = thr(iid + "/thr", thrkey, ["C1", thrkey], rr)
        v = verdict("C1", rr) if label == "u1fcn minus strongest rule" else ""
        emit(["", text, tex(fmt_num(a, "f4")), f"{tex(fmt_num(b, 'f4'))} {ci_txt(lo, hi)}", f"$\\geq$ {t:.3f}", v])
    # every seed above the rule
    ln_seed = row_line(H, head, "u1fcn per seed")
    ln_rule = row_line(H, head, "Strongest rule (Table times teammates over opponents (a=2, b=0.5))")
    s25 = [float(x) for x in md_row(H, ln_seed)["24/25"].split()]
    s26 = [float(x) for x in md_row(H, ln_seed)["Held-out"].split()]
    r25 = float(md_row(H, ln_rule)["24/25"]); r26 = float(md_row(H, ln_rule)["Held-out"])
    k25, k26 = sum(x > r25 for x in s25), sum(x > r26 for x in s26)
    rr = r[0]
    R.add("C1/seeds_above/2425", k25, dict(kind="derived", formula=f"L{ln_seed} column 24/25, number of the three seed values {s25} above the rule of L{ln_rule} ({r25})", inputs=[]),
          display=f"{k25} of 3", cell=[rr, 2])
    R.add("C1/seeds_above/2526", k26, dict(kind="derived", formula=f"L{ln_seed} column Held-out, number of the three seed values {s26} above the rule of L{ln_rule} ({r26})", inputs=[]),
          display=f"{k26} of 3", cell=[rr, 3])
    emit(["", "Seeds of the variant above the rule", f"{k25} of 3", f"{k26} of 3", "all 3", ""])
    rows.append("\\addlinespace")

    # ---------------- C2
    rows.append("\\multicolumn{6}{l}{\\textit{C2. Enriched scalars replace most of the spatial gain on outcome tasks (mean over Pass, Ball recovery, Pressure)}} \\\\")
    head = "## C2"
    spec = [  # (md row label, column, id, json path, English label, threshold key, threshold prefix)
        ("XGB enriched minus M4 thin, held-out", "Three-task mean", "C2/a_xgbrich_minus_m4thin_mean3", ["derived", "C2", "xgbrich_minus_m4thin_mean3"],
         "(a) Enriched XGBoost minus thin-scalar CNN", "xgbrich_minus_m4thin_confirm", "$\\geq$"),
        ("Enriched gain M4 minus M2, held-out", "Three-task mean", "C2/b_rich_gain_mean3", ["derived", "C2", "rich_gain_mean3"],
         "(b) Enriched gain, CNN minus MLP with enriched scalars", "rich_gain_abs_confirm", "$\\leq$"),
        ("Thin gain M4 minus M2, held-out", "Three-task mean", "C2/thin_gain_mean3", ["derived", "C2", "thin_gain_mean3"],
         "\\phantom{(b)} Same gain with thin scalars", None, None),
        ("Thin gain M4 minus M2, held-out", "pressure", "C2/c_pressure_thin_gain", ["derived", "C2", "thin_gain", "pressure"],
         "(c) Pressure, gain with thin scalars", "pressure_thin_gain_min", "$\\geq$"),
        ("Enriched gain M4 minus M2, held-out", "pressure", "C2/c_pressure_rich_gain", ["derived", "C2", "rich_gain", "pressure"],
         "\\phantom{(c)} Pressure, enriched gain", "pressure_rich_gain_max", "$\\leq$"),
        ("XGB enriched minus M4 thin, held-out", "pressure", "C2/report_pressure_xgbrich_minus_m4thin", ["derived", "C2", "xgbrich_minus_m4thin", "pressure"],
         "Reported only, Pressure, enriched XGBoost minus thin-scalar CNN", None, None),
    ]
    def ratio_row():
        ln = row_line(H, head, "Enriched gain mean over thin gain mean")
        cell = md_row(H, ln)["Three-task mean"]
        m = re.match(r"^([0-9.]+) \(([0-9.]+) in 24/25\)$", cell)
        q26, q25 = float(m.group(1)), float(m.group(2))
        rr = r[0]
        R.add("C2/b_ratio/2526", q26, dict(kind="mdre", file=rel(H), line=ln, col="Three-task mean", regex=r"^([0-9.]+) \(", group=1), display=f"{q26:.2f}", cell=[rr, 3])
        R.add("C2/b_ratio/2425", q25, dict(kind="mdre", file=rel(H), line=ln, col="Three-task mean", regex=r"\(([0-9.]+) in 24/25\)", group=1), display=f"{q25:.2f}", cell=[rr, 2])
        assert round(json_value(J, ["derived", "C2", "rich_over_thin_ratio3"]), 2) == q26
        t = thr("C2/b_ratio/thr", "rich_gain_ratio_confirm", ["C2", "rich_gain_ratio_confirm"], rr)
        emit(["", "\\phantom{(b)} Enriched gain over thin gain", f"{q25:.2f}", f"{q26:.2f}", f"$\\leq$ {t:.3f}", ""])


    first = True
    for label, col, iid, jp, text, thrkey, pre in spec:
        ln = row_line(H, head, label)
        rr = r[0]
        b = R.md(iid + "/2526", H, ln, col, fmt="f4", cell=[rr, 3])
        lo = R.md(iid + "/2526_lo", H, ln + 1, col, "lo", fmt="f4", cell=[rr, 3])
        hi = R.md(iid + "/2526_hi", H, ln + 1, col, "hi", fmt="f4", cell=[rr, 3])
        assert md_row(H, ln + 1)["Quantity"] == "Same, 95% CI" and md_row(H, ln + 2)["Quantity"] == "Same, 24/25"
        a = R.md(iid + "/2425", H, ln + 2, col, fmt="f4", cell=[rr, 2])
        check(b, jp)
        cjp = ["ci", "derived", "C2"] + jp[2:]
        check(lo, cjp + [0]); check(hi, cjp + [1])
        if thrkey:
            t = thr(iid + "/thr", thrkey, ["C2", thrkey], rr)
            tt = f"{pre} {tex(fmt_num(t, 'f3'))}"
            if thrkey == "rich_gain_abs_confirm":
                tt += " and $\\leq$ half of thin gain"
        else:
            tt = "not judged" if "Reported" in text else "reference for (b)"
        v = verdict("C2", rr) if first else ""
        first = False
        emit(["", text, tex(fmt_num(a, "f4")), f"{tex(fmt_num(b, 'f4'))} {ci_txt(lo, hi)}", tt, v])
        if iid == "C2/thin_gain_mean3":
            ratio_row()
    rows.append("\\addlinespace")

    # ---------------- C3
    rows.append("\\multicolumn{6}{l}{\\textit{C3. The unified model stays close to enriched XGBoost (unified minus enriched XGBoost, AUC)}} \\\\")
    head = "## C3"
    ln = row_line(H, head, "u1 minus XGB enriched, held-out")
    assert md_row(H, ln + 1)["Quantity"] == "Same, 95% CI" and md_row(H, ln + 2)["Quantity"] == "Same, 24/25"
    for k, (col, jp, text, thrkey) in enumerate([
            ("pass", ["pass"], "Pass", "pass_abs_confirm"),
            ("pressure", ["pressure"], "Pressure", "pressure_abs_confirm"),
            ("ball_recovery", ["ball_recovery"], "Ball recovery", "br_abs_confirm"),
            ("Three-task mean (Pass BR Pressure)", None, "Mean of the three tasks", "mean3_abs_confirm")]):
        rr = r[0]
        iid = f"C3/{thrkey.replace('_abs_confirm', '')}"
        b = R.md(iid + "/2526", H, ln, col, fmt="f4", cell=[rr, 3])
        lo = R.md(iid + "/2526_lo", H, ln + 1, col, "lo", fmt="f4", cell=[rr, 3])
        hi = R.md(iid + "/2526_hi", H, ln + 1, col, "hi", fmt="f4", cell=[rr, 3])
        a = R.md(iid + "/2425", H, ln + 2, col, fmt="f4", cell=[rr, 2])
        if jp:
            check(b, ["derived", "C3", "u1_minus_xgbrich"] + jp)
            check(lo, ["ci", "derived", "C3", "u1_minus_xgbrich"] + jp + [0]); check(hi, ["ci", "derived", "C3", "u1_minus_xgbrich"] + jp + [1])
        else:
            check(b, ["derived", "C3", "u1_minus_xgbrich_mean3"])
            check(lo, ["ci", "derived", "C3", "u1_minus_xgbrich_mean3", 0]); check(hi, ["ci", "derived", "C3", "u1_minus_xgbrich_mean3", 1])
        t = thr(iid + "/thr", thrkey, ["C3", thrkey], rr)
        v = verdict("C3", rr) if k == 0 else ""
        emit(["", text, tex(fmt_num(a, "f4")), f"{tex(fmt_num(b, 'f4'))} {ci_txt(lo, hi)}", f"$|d| \\leq$ {t:.3f}", v])
    rows.append("\\addlinespace")

    # ---------------- C4
    rows.append("\\multicolumn{6}{l}{\\textit{C4. The destination gate pattern holds on new matches (gate reading, no retraining)}} \\\\")
    for k, (tag, n_seed, text, rk25_expect, gkey_min, gkey_rank, pre) in enumerate([
            ("u1", 4, "Unified model, destination gate, four seeds", "1 1 1 1", "u1_dest_gate_min", "u1_dest_rank", "$\\geq$"),
            ("shuf_dest", 3, "Destination-shuffled model, destination gate, three seeds", "8 8 8", "shuf_dest_gate_max", "shuf_dest_rank", "$\\leq$")]):
        rr = r[0]
        lu = md_find(U, f"## Run {tag} (", 0, task="dest")
        # the 2024/25 minimum and maximum are the unrounded per-seed values rounded once to three decimals; gate_min and gate_max
        # of the summary are already rounded to four decimals and rounding them again would turn 0.27847 into 0.279, so the summary is only a check
        per25 = unified_seed_values(tag, "dest", "gate")
        assert sorted(per25) == list(range(n_seed)), f"{tag} 24/25 seeds {sorted(per25)}"
        for s, v in per25.items():
            R.perevent(f"C4/{tag}/2425/s{s}", v, f"mean gate_mean of the dest events of {tag}_s{s}.npz, 202 matches", fmt="f4")
        ids25 = [f"C4/{tag}/2425/s{s}" for s in per25]
        assert round(min(per25.values()), 4) == md_value(U, lu, "gate_min") and round(max(per25.values()), 4) == md_value(U, lu, "gate_max"), \
            f"{tag} minimum or maximum of the per-seed gates differs from L{lu} of the summary table"
        gmin = R.derived(f"C4/{tag}/2425_min", min(per25.values()), "minimum over these seeds", ids25, fmt="f3", cell=[rr, 2])
        gmax = R.derived(f"C4/{tag}/2425_max", max(per25.values()), "maximum over these seeds", ids25, fmt="f3", cell=[rr, 2])
        rk = md_row(U, lu)["gate_rank_by_seed"]
        assert rk == rk25_expect, f"{tag} 24/25 ranks {rk}"
        R.add(f"C4/{tag}/2425_rank", int(rk.split()[0]), dict(kind="md", file=rel(U), line=lu, col="gate_rank_by_seed", part="text"),
              display=f"rank {rk.split()[0]} of 8", cell=[rr, 2])
        seeds = sorted(json_value(J, ["derived", "C4", tag]).keys())
        g26 = []
        for s in seeds:
            g26.append(R.js(f"C4/{tag}/2526_s{s}", J, ["derived", "C4", tag, s, "dest_gate"], fmt="f3"))
            rank = json_value(J, ["derived", "C4", tag, s, "dest_rank"])
            assert rank == (1 if tag == "u1" else 6)
        assert len(seeds) == n_seed
        R.add(f"C4/{tag}/2526_rank", 1 if tag == "u1" else 6, dict(kind="json", file=rel(J), path=["derived", "C4", tag, "*", "dest_rank"]),
              display=f"rank {1 if tag == 'u1' else 6} of 6", cell=[rr, 3])
        lo26, hi26 = min(g26), max(g26)
        ids = [f"C4/{tag}/2526_s{s}" for s in seeds]
        R.derived(f"C4/{tag}/2526_min", lo26, "minimum over these seeds", ids, fmt="f3", cell=[rr, 3])
        R.derived(f"C4/{tag}/2526_max", hi26, "maximum over these seeds", ids, fmt="f3", cell=[rr, 3])
        t = thr(f"C4/{tag}/thr", gkey_min, ["C4", gkey_min], rr)
        tr = "rank 1" if tag == "u1" else "last rank"
        v = verdict("C4", rr) if k == 0 else ""
        emit(["", text, f"{gmin:.3f} to {gmax:.3f}, rank {rk.split()[0]} of 8", f"{lo26:.3f} to {hi26:.3f}, rank {1 if tag == 'u1' else 6} of 6",
              f"{pre} {t:.3f}, {tr}", v])

    get = lambda i: next(it["value"] for it in R.items if it["id"] == i)
    R.derived("note/pressure_rich_gain", get("C2/c_pressure_rich_gain/2526"), "equal to C2/c_pressure_rich_gain/2526", ["C2/c_pressure_rich_gain/2526"], fmt="f4", cell=["note", 0])
    R.derived("note/pressure_bound", get("C2/c_pressure_rich_gain/thr"), "equal to C2/c_pressure_rich_gain/thr", ["C2/c_pressure_rich_gain/thr"], fmt="f3", cell=["note", 0])
    body = ["\\toprule",
            " & Quantity & 2024/25 & 2025/26 [95\\% CI] & Threshold to confirm & Verdict \\\\",
            "\\midrule"] + rows + ["\\bottomrule"]
    note = ("The four claims, the quantities and the thresholds were written down and frozen before any model was evaluated on the 146 "
            "matches of 2025/26, and the evaluation was run once. Each claim also had a threshold for not confirmed, written at the same time; results "
            "between the two thresholds are inconclusive. Each claim is judged on its own. Values are differences in top-1 (C1) or AUC (C2, C3) "
            "and gate values (C4); 2024/25 is the 202-match test set. The 95\\% CI is a bootstrap interval over matches (2000 resamples). AUC is the area under the receiver operating characteristic curve. "
            "The variant in C1 is the resolution-preserving unified model, averaged over three seeds. C2 and C3 use the mean over Pass, Ball "
            "recovery and Pressure, the three outcome tasks with enough held-out events; Shot and Interception were reported but not judged, "
            "and Dribble and Tackle were not part of the held-out set. In C2, the gain is CNN (concatenation network with a convolutional branch) minus MLP (multilayer perceptron), both single-task, with thin or enriched "
            "scalars; the enriched gain on Pressure of 0.0091 is close to its bound of 0.010. In C3, $d$ is the unified model (pooled encoder, "
            "four seeds) minus enriched XGBoost (five seeds). C4 applies the trained gates to the new events without retraining, so it checks "
            "that the gate pattern holds and does not test the closing of the gate itself; ranks are among the eight tasks in 2024/25 and among "
            "the six held-out tasks in 2025/26, and the gate readings carry no bootstrap interval.")
    caption = "Confirmation on the 146 held-out matches of 2025/26."
    write_table(NAME, body, caption, "tab:holdout", env="table*", note=note, tabular="tabularx", width="\\linewidth",
                colspec="l >{\\raggedright\\arraybackslash}X l l >{\\raggedright\\arraybackslash}p{2.6cm} >{\\raggedright\\arraybackslash}p{1.9cm}")
    R.save()
    print(f"[{NAME}] {len(R.items)} numbers written")


if __name__ == "__main__":
    main()
