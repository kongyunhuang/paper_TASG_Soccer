#!/usr/bin/env python3
"""
表 tab_holdout  25/26 的 146 场留出确认，四条结论各自的量、24/25 和 25/26 的值、确认阈值和判定
============================
量和区间取 audit/holdout_2526_summary.md（C1 到 C4 四节，按行号登记），阈值取同一文件开头的阈值字典（和 summary.json 的 rules 相同），
判定词取冻结方案文末“判定”表（memory_logs/2026-10-02_留出数据确认_冻结方案和判定标准.md），照搬，不写总括句。
C4 的 24/25 门控最小、最大取 u1_clean 逐事件文件的逐种子原值再舍成三位（和 u1_clean_summary.md“运行 u1”“运行 shuf_dest”两表的
gate_min、gate_max 四位核对，10-06 起不再拿四位的汇总值直接舍成三位），名次取这两张表；25/26 取 summary.json 的 derived.C4。
每个 25/26 的数另和 summary.json 的 derived 字段核对，四舍五入到四位一致才写表。

输入  audit/holdout_2526_summary.md，data/cache/holdout_2526/summary.json，audit/u1_clean_summary.md，data/cache/u1_clean/{u1,shuf_dest}_s*.npz，冻结方案文件
输出  outputs/tables/tab_holdout.tex，scripts/paper_v2/out/tab_holdout_sources.json

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/paper_v2/tab_holdout.py

Last modified 2026-10-06（C4 的 24/25 门控范围改从逐种子原值舍入，shuf_dest 最大值由 0.279 改为 0.278；表注 C2 句措辞按 Miguel 10-05 修订）
"""
import re

from scripts.paper_v2.common import (SRC, Registry, fmt_num, json_value, md_find, md_row, md_value, rel, tex, unified_seed_values, write_table)

NAME = "tab_holdout"
H, J, U, F = SRC["hold_md"], SRC["hold_json"], SRC["u1"], SRC["frozen"]


def lines(p):
    return p.read_text(encoding="utf8").split("\n")


def row_line(path, head, label):
    """在 head 一节的表里找第一列等于 label 的行号"""
    return md_find(path, head, 0, **{"量": label})


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
        assert round(jv, 4) == round(v, 4), f"{jpath} json {jv} 和 md {v} 不一致"

    # 判定词，冻结方案文末判定表
    FL = lines(F)
    v0 = next(i for i, x in enumerate(FL) if x.startswith("### 判定（按第二节的阈值）"))
    verdict_line = {c: next(i for i, x in enumerate(FL) if i > v0 and x.startswith(f"| {c} ")) + 1 for c in ["C1", "C2", "C3", "C4"]}
    VERD = {"C1": ("Confirmed", "**确认**"), "C2": ("Confirmed, Pressure near the bound", "**确认**"),
            "C3": ("Inconclusive", "**不确定**"), "C4": ("Holds", "**保持**")}

    def verdict(c, rr):
        txt, must = VERD[c]
        R.text(f"{c}/verdict", txt, F, verdict_line[c], must, cell=[rr, 5])
        if c == "C2":
            R.text("C2/verdict_nearline", "near the bound", F, verdict_line[c], "贴着 0.010 的线", cell=[rr, 5])
        return txt

    # 阈值字典的行号（holdout md 开头的代码块）
    HL = lines(H)

    def thr_line(key):
        return next(i for i, x in enumerate(HL) if f'"{key}"' in x) + 1

    def thr(iid, key, cpath, rr):
        ln = thr_line(key)
        v = float(re.search(r":\s*([-0-9.]+)", HL[ln - 1]).group(1))
        assert v == json_value(J, ["rules"] + cpath), f"{key} 阈值和 json 不一致"
        R.add(iid, v, dict(kind="mdline", file=rel(H), line=ln, regex=r":\s*([-0-9.]+)"), display=fmt_num(v, "f3"), cell=[rr, 4], fmt="f3")
        return v

    # ---------------- C1
    rows.append("\\multicolumn{6}{l}{\\textit{C1. The spatial structure cannot be replaced for pass destination}} \\\\")
    head = "## C1"
    for label, iid, thrkey, jkey, text in [
        ("u1fcn 减最强规则", "C1/fcn_minus_rule", "fcn_minus_rule_confirm", "fcn_minus_rule", "Variant minus strongest rule, top-1"),
        ("u1fcn 减 M2 加厚", "C1/fcn_minus_m2rich", "fcn_minus_m2rich_confirm", "fcn_minus_m2rich", "Variant minus enriched-scalar MLP, top-1")]:
        ln = row_line(H, head, label)
        rr = r[0]
        a = R.md(iid + "/2425", H, ln, "24/25", fmt="f4", cell=[rr, 2])
        b = R.md(iid + "/2526", H, ln, "留出集", fmt="f4", cell=[rr, 3])
        lo = R.md(iid + "/2526_lo", H, ln, "95% 区间", "lo", fmt="f4", cell=[rr, 3])
        hi = R.md(iid + "/2526_hi", H, ln, "95% 区间", "hi", fmt="f4", cell=[rr, 3])
        check(b, ["derived", "C1", jkey])
        check(lo, ["ci", "derived", "C1", jkey, 0]); check(hi, ["ci", "derived", "C1", jkey, 1])
        t = thr(iid + "/thr", thrkey, ["C1", thrkey], rr)
        v = verdict("C1", rr) if label == "u1fcn 减最强规则" else ""
        emit(["", text, tex(fmt_num(a, "f4")), f"{tex(fmt_num(b, 'f4'))} {ci_txt(lo, hi)}", f"$\\geq$ {t:.3f}", v])
    # 逐种子都高于规则
    ln_seed = row_line(H, head, "u1fcn 逐种子")
    ln_rule = row_line(H, head, "最强规则（条件表乘队友再除对手（a=2，b=0.5））")
    s25 = [float(x) for x in md_row(H, ln_seed)["24/25"].split()]
    s26 = [float(x) for x in md_row(H, ln_seed)["留出集"].split()]
    r25 = float(md_row(H, ln_rule)["24/25"]); r26 = float(md_row(H, ln_rule)["留出集"])
    k25, k26 = sum(x > r25 for x in s25), sum(x > r26 for x in s26)
    rr = r[0]
    R.add("C1/seeds_above/2425", k25, dict(kind="derived", formula=f"L{ln_seed} 列 24/25 的三个种子值 {s25} 里大于 L{ln_rule} 规则 {r25} 的个数", inputs=[]),
          display=f"{k25} of 3", cell=[rr, 2])
    R.add("C1/seeds_above/2526", k26, dict(kind="derived", formula=f"L{ln_seed} 列 留出集 的三个种子值 {s26} 里大于 L{ln_rule} 规则 {r26} 的个数", inputs=[]),
          display=f"{k26} of 3", cell=[rr, 3])
    emit(["", "Seeds of the variant above the rule", f"{k25} of 3", f"{k26} of 3", "all 3", ""])
    rows.append("\\addlinespace")

    # ---------------- C2
    rows.append("\\multicolumn{6}{l}{\\textit{C2. Enriched scalars replace most of the spatial gain on outcome tasks (mean over Pass, Ball recovery, Pressure)}} \\\\")
    head = "## C2"
    spec = [  # (md 行名, 列, id, json 路径, 英文, 阈值键, 阈值显示前缀)
        ("XGB 加厚减 M4 薄 留出集", "三任务平均", "C2/a_xgbrich_minus_m4thin_mean3", ["derived", "C2", "xgbrich_minus_m4thin_mean3"],
         "(a) Enriched XGBoost minus thin-scalar CNN", "xgbrich_minus_m4thin_confirm", "$\\geq$"),
        ("厚增益 M4 减 M2 留出集", "三任务平均", "C2/b_rich_gain_mean3", ["derived", "C2", "rich_gain_mean3"],
         "(b) Enriched gain, CNN minus MLP with enriched scalars", "rich_gain_abs_confirm", "$\\leq$"),
        ("薄增益 M4 减 M2 留出集", "三任务平均", "C2/thin_gain_mean3", ["derived", "C2", "thin_gain_mean3"],
         "\\phantom{(b)} Same gain with thin scalars", None, None),
        ("薄增益 M4 减 M2 留出集", "pressure", "C2/c_pressure_thin_gain", ["derived", "C2", "thin_gain", "pressure"],
         "(c) Pressure, gain with thin scalars", "pressure_thin_gain_min", "$\\geq$"),
        ("厚增益 M4 减 M2 留出集", "pressure", "C2/c_pressure_rich_gain", ["derived", "C2", "rich_gain", "pressure"],
         "\\phantom{(c)} Pressure, enriched gain", "pressure_rich_gain_max", "$\\leq$"),
        ("XGB 加厚减 M4 薄 留出集", "pressure", "C2/report_pressure_xgbrich_minus_m4thin", ["derived", "C2", "xgbrich_minus_m4thin", "pressure"],
         "Reported only, Pressure, enriched XGBoost minus thin-scalar CNN", None, None),
    ]
    def ratio_row():
        ln = row_line(H, head, "厚增益平均 对 薄增益平均")
        cell = md_row(H, ln)["三任务平均"]
        m = re.match(r"^([0-9.]+)（24/25 是 ([0-9.]+)）$", cell)
        q26, q25 = float(m.group(1)), float(m.group(2))
        rr = r[0]
        R.add("C2/b_ratio/2526", q26, dict(kind="mdre", file=rel(H), line=ln, col="三任务平均", regex=r"^([0-9.]+)（", group=1), display=f"{q26:.2f}", cell=[rr, 3])
        R.add("C2/b_ratio/2425", q25, dict(kind="mdre", file=rel(H), line=ln, col="三任务平均", regex=r"24/25 是 ([0-9.]+)）", group=1), display=f"{q25:.2f}", cell=[rr, 2])
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
        assert md_row(H, ln + 1)["量"] == "同上 95% 区间" and md_row(H, ln + 2)["量"] == "同上 24/25"
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
    ln = row_line(H, head, "u1 减 XGB 加厚 留出集")
    assert md_row(H, ln + 1)["量"] == "同上 95% 区间" and md_row(H, ln + 2)["量"] == "同上 24/25"
    for k, (col, jp, text, thrkey) in enumerate([
            ("pass", ["pass"], "Pass", "pass_abs_confirm"),
            ("pressure", ["pressure"], "Pressure", "pressure_abs_confirm"),
            ("ball_recovery", ["ball_recovery"], "Ball recovery", "br_abs_confirm"),
            ("三任务平均（Pass BR Pressure）", None, "Mean of the three tasks", "mean3_abs_confirm")]):
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
        lu = md_find(U, f"## 运行 {tag}（", 0, task="dest")
        # 24/25 的最小、最大取逐事件文件的逐种子原值再舍成三位。汇总表的 gate_min、gate_max 已舍成四位，再舍一次会把
        # shuf_dest 种子 2 的 0.27847 舍成 0.279（10-06 改），所以汇总表只拿来核对，四位一致才写表
        per25 = unified_seed_values(tag, "dest", "gate")
        assert sorted(per25) == list(range(n_seed)), f"{tag} 24/25 种子 {sorted(per25)}"
        for s, v in per25.items():
            R.perevent(f"C4/{tag}/2425/s{s}", v, f"{tag}_s{s}.npz 落点事件的 gate_mean 平均，202 场", fmt="f4")
        ids25 = [f"C4/{tag}/2425/s{s}" for s in per25]
        assert round(min(per25.values()), 4) == md_value(U, lu, "gate_min") and round(max(per25.values()), 4) == md_value(U, lu, "gate_max"), \
            f"{tag} 逐种子门控的最小、最大和汇总表 L{lu} 对不上"
        gmin = R.derived(f"C4/{tag}/2425_min", min(per25.values()), "这几个种子里的最小值", ids25, fmt="f3", cell=[rr, 2])
        gmax = R.derived(f"C4/{tag}/2425_max", max(per25.values()), "这几个种子里的最大值", ids25, fmt="f3", cell=[rr, 2])
        rk = md_row(U, lu)["gate_rank_by_seed"]
        assert rk == rk25_expect, f"{tag} 24/25 名次 {rk}"
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
        R.derived(f"C4/{tag}/2526_min", lo26, "这几个种子里的最小值", ids, fmt="f3", cell=[rr, 3])
        R.derived(f"C4/{tag}/2526_max", hi26, "这几个种子里的最大值", ids, fmt="f3", cell=[rr, 3])
        t = thr(f"C4/{tag}/thr", gkey_min, ["C4", gkey_min], rr)
        tr = "rank 1" if tag == "u1" else "last rank"
        v = verdict("C4", rr) if k == 0 else ""
        emit(["", text, f"{gmin:.3f} to {gmax:.3f}, rank {rk.split()[0]} of 8", f"{lo26:.3f} to {hi26:.3f}, rank {1 if tag == 'u1' else 6} of 6",
              f"{pre} {t:.3f}, {tr}", v])

    get = lambda i: next(it["value"] for it in R.items if it["id"] == i)
    R.derived("note/pressure_rich_gain", get("C2/c_pressure_rich_gain/2526"), "同 C2/c_pressure_rich_gain/2526", ["C2/c_pressure_rich_gain/2526"], fmt="f4", cell=["note", 0])
    R.derived("note/pressure_bound", get("C2/c_pressure_rich_gain/thr"), "同 C2/c_pressure_rich_gain/thr", ["C2/c_pressure_rich_gain/thr"], fmt="f3", cell=["note", 0])
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
    print(f"[{NAME}] 写好 {len(R.items)} 个数")


if __name__ == "__main__":
    main()
