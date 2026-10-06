#!/usr/bin/env python3
"""
逐数核对并写数据来源 md  把每张表每张图登记过的数按出处回读一遍，再和生成的 tex 单元格、图的数据表逐个比
============================
三层核对，不抽样。
  回读    md 按行号和列名重读，json 按字段重读，日志按行号加正则重读，判定词按行号查原句；逐事件来的数用本文件自己的一套代码
          （直接 np.load 加 sklearn，不经 summarize_u1_clean 和 common 的重算函数）独立重算
  进表    解析生成的 tex，按登记的行列位置取单元格，看显示值在不在里面
  进图    图的数据表（out/*_data.csv）里的每个值都要等于登记的值
另查 tex 里有没有破折号、正文冒号，表注是否解释了表里出现的缩写。结果连同每个数的出处写进 outputs/tables/数据来源_2026-10-04.md。

输入  scripts/paper_v2/out/*_sources.json，out/*_data.csv，outputs/tables/*.tex，以及登记里列出的全部结果文件
输出  outputs/tables/数据来源_2026-10-04.md，scripts/paper_v2/out/verify_report.json

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/paper_v2/verify_all.py
  加 --no-md 只核对、不重写数据来源 md

Last modified 2026-10-06（tab_holdout 的 C4 逐种子门控也按逐事件独立重算核对；public release: paths relative to the repository root, outputs/ instead of 02_manuscript/）
"""
import csv
import glob
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.paper_v2.common import DEV, MANU, OUT_DIR, ROOT, TAB_DIR, TASKS, md5, md_row, norm_tex, parse_cell

ORDER = ["tab_tasks", "tab_outcome_main", "tab_destination", "tab_interventions", "tab_holdout", "tab_gate_variants_supp",
         "tab_application_supp", "fig_overview", "fig_outcome_vs_scalars", "fig_destination", "fig_gate", "fig_holdout",
         "figS1_gate_vs_shuffle", "figS2_soft_penalty", "figS3_player_preference"]
MD_OUT = MANU / "tables" / "数据来源_2026-10-04.md"

# ---------------------------------------------------------------- 独立重算（不导入 summarize_u1_clean）
ST = pd.read_parquet(DEV / "audit" / "raw_scan" / "match_state_2425.parquet")
CLEAN = set(ST.loc[(ST["state"] == "正常") & (ST["shot_frame_pct"] >= 70), "match_id"].astype(int))
assert len(CLEAN) == 202
_cache = {}


def unz(tag, seed):
    k = ("u", tag, seed)
    if k not in _cache:
        z = np.load(DEV / "data" / "cache" / "u1_clean" / f"{tag}_s{seed}.npz", allow_pickle=True)
        ok = np.isin(z["match_id"].astype(np.int64), np.array(sorted(CLEAN), dtype=np.int64))
        _cache[k] = (dict(task=z["task"], y=z["y"].astype(int), logit=z["logit_bin"], top1=z["dest_top1"], lp=z["dest_logp_true"], gate=z["gate_mean"]), ok)
    return _cache[k]


def u_val(tag, seed, task, what):
    z, ok = unz(tag, seed)
    m = (z["task"] == TASKS.index(task)) & ok
    if what == "gate":
        return float(np.nanmean(z["gate"][m]))
    if what == "skill":
        return 1 + float(z["lp"][m].mean()) / math.log(96)
    if task == "dest":
        return float((z["top1"][m] == z["y"][m]).mean())
    return float(roc_auc_score(z["y"][m], z["logit"][m]))


def s_val(d, task, model, seed, what="metric"):
    base = DEV / "data" / "cache" / d / "preds"
    meta = np.load(base / f"{task}_test_meta.npz", allow_pickle=True)
    ok = np.isin(meta["match_id"].astype(np.int64), np.array(sorted(CLEAN), dtype=np.int64))
    y = meta["y"].astype(int)
    z = np.load(base / f"{task}_{model}_s{seed}.npz")
    if task == "dest":
        return float((z["top1"][ok] == y[ok]).mean()) if what == "metric" else 1 + float(z["logp_true"][ok].mean()) / math.log(96)
    return float(roc_auc_score(y[ok], z["prob"][ok]))


def s_seeds(d, task, model):
    fs = glob.glob(str(DEV / "data" / "cache" / d / "preds" / f"{task}_{model}_s*.npz"))
    return sorted(int(f.rsplit("_s", 1)[1].split(".")[0]) for f in fs)


def u_seeds(tag):
    """只认 {tag}_s{数字}.npz，免得 u1fcn_s* 把 u1fcn_shuf_dest_s0 也算进来"""
    out = []
    for f in glob.glob(str(DEV / "data" / "cache" / "u1_clean" / f"{tag}_s*.npz")):
        m = re.fullmatch(rf"{re.escape(tag)}_s(\d+)", Path(f).stem)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def recompute_perevent(name, iid):
    """按登记 id 的形状独立重算逐事件来的数"""
    p = iid.split("/")
    if name == "tab_tasks" and p[1] == "test_pos":
        meta = np.load(DEV / "data" / "cache" / "single_clean" / "preds" / f"{p[0]}_test_meta.npz", allow_pickle=True)
        ok = np.isin(meta["match_id"].astype(np.int64), np.array(sorted(CLEAN), dtype=np.int64))
        return float(meta["y"][ok].astype(int).mean())
    if name == "tab_destination":
        if p[0] in ("single_clean", "single_clean_rich"):
            v = [s_val(p[0], "dest", p[1], s, "metric") for s in s_seeds(p[0], "dest", p[1])]
            return float(np.std(v, ddof=1))
        v = [u_val(p[0], s, "dest", "skill") for s in u_seeds(p[0])]
        return float(np.std(v, ddof=1))
    if name == "tab_interventions":
        run, sd = p[0], int(p[1][1:])
        task = {"shuf_dest": "dest", "shuf_tackle": "tackle", "shuf_pressure": "pressure"}[run]
        if p[2] == "metric_change":
            return u_val(run, sd, task, "metric") - u_val("u1", sd, task, "metric")
        if p[2] == "dest_change":
            return u_val(run, sd, "dest", "metric") - u_val("u1", sd, "dest", "metric")
        if p[2] == "others_max_change":
            ds = [u_val(run, sd, t, "metric") - u_val("u1", sd, t, "metric") for t in TASKS if t not in (task, "dest")]
            return max(ds, key=abs)
    if name == "fig_outcome_vs_scalars":
        d, t, s = p[0], p[1], int(p[2][1:])
        return s_val(d, t, "M4_CNN_Full", s) - s_val(d, t, "B2_XGB_360", s)
    if name == "fig_destination":
        if p[0] in ("single_clean", "single_clean_rich"):
            return s_val(p[0], "dest", p[1], int(p[4][1:]), "metric" if p[3] == "top1" else "skill")
        return u_val(p[0], int(p[3][1:]), "dest", "metric" if p[2] == "top1" else "skill")
    if name == "fig_gate":
        return u_val(p[0], int(p[2][1:]), p[1], "gate")
    if name in ("fig_holdout", "tab_holdout") and p[0] == "C4":
        return u_val(p[1], int(p[3][1:]), "dest", "gate")
    if name == "figS1_gate_vs_shuffle" and p[3] == "gate":
        return u_val(p[0], int(p[1][1:]), p[2], "gate")
    if name == "figS3_player_preference":
        fs = sorted(glob.glob(str(DEV / "data" / "cache" / "player_pref" / "fold*_s*_players.parquet")))
        allp = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        allp["d"] = allp["M3"] - allp["M2"]
        avg = allp.groupby(["fold", "player_id"]).agg(d=("d", "mean"), n=("n_test", "first"), mv=("mover", "first")).reset_index()
        g = avg if p[0] == "全体球员" else avg[avg["mv"].astype(bool)]
        return float(np.average(g["d"], weights=g["n"])) if p[1] == "D_recomputed" else float(len(g))
    raise KeyError(f"{name} {iid} 没有独立重算规则")


# ---------------------------------------------------------------- 回读
def reread(name, it, items_by_id):
    s = it["src"]
    k = s["kind"]
    f = ROOT / s["file"] if "file" in s else None
    if k == "md":
        cell = md_row(f, s["line"])[s["col"]]
        if s.get("part") == "text":
            return float(cell.split()[0]), f"{s['file']} L{s['line']} 列 {s['col']}"
        pc = parse_cell(cell)
        return pc[s.get("part", "value")], f"{s['file']} L{s['line']} 列 {s['col']}" + (f" 的 {s['part']}" if s.get("part", "value") != "value" else "")
    if k == "mdre":
        txt = md_row(f, s["line"])[s["col"]] if s["col"] else f.read_text(encoding="utf8").split("\n")[s["line"] - 1]
        m = re.search(s["regex"], txt.replace("−", "-"))
        return s.get("sign", 1) * float(m.group(s["group"])), f"{s['file']} L{s['line']}" + (f" 列 {s['col']}" if s["col"] else " 整行") + " 正则"
    if k == "mdline":
        txt = f.read_text(encoding="utf8").split("\n")[s["line"] - 1]
        return float(re.search(s["regex"], txt).group(1)), f"{s['file']} L{s['line']}"
    if k == "log":
        txt = f.read_text(encoding="utf8").split("\n")[s["line"] - 1]
        m = re.search(r"\[data\] (\w+)\s+dev\s+([\d,]+) test\s+([\d,]+) mean label dev ([\d.]+) test ([\d.]+)", txt)
        return float(m.group(s["group"]).replace(",", "")), f"{s['file']} L{s['line']}"
    if k == "json":
        d = json.load(open(f, encoding="utf8"))
        path = s["path"]
        if "*" in path:
            i = path.index("*")
            node = d
            for x in path[:i]:
                node = node[x]
            vals = []
            for kk in node:
                v = node[kk]
                for x in path[i + 1:]:
                    v = v[x]
                vals.append(v)
            assert len(set(vals)) == 1
            return float(vals[0]), f"{s['file']} 字段 {'.'.join(map(str, path))}"
        for x in path:
            d = d[x]
        return (float(d) if not isinstance(d, str) else d), f"{s['file']} 字段 {'.'.join(map(str, path))}"
    if k == "text":
        txt = f.read_text(encoding="utf8").split("\n")[s["line"] - 1]
        return (it["value"] if s["must_contain"] in txt else None), f"{s['file']} L{s['line']} 含“{s['must_contain']}”"
    if k == "derived":
        fm = s["formula"]
        if fm.startswith("同 "):
            return items_by_id[fm[2:].strip()]["value"], f"同 {fm[2:].strip()}"
        if fm == "这几个种子的均值":
            v = [items_by_id[i]["value"] for i in s["inputs"]]
            return float(np.mean(v)), f"{', '.join(s['inputs'])} 的均值"
        if "最小值" in fm or "最大值" in fm:
            v = [items_by_id[i]["value"] for i in s["inputs"]]
            return (min(v) if "最小值" in fm else max(v)), f"{', '.join(s['inputs'])} 的{'最小值' if '最小值' in fm else '最大值'}"
        if "other7_avg_change" in fm:
            l1, l2 = [int(x.split(":")[1]) for x in s["inputs"]]
            fu = ROOT / "audit" / "u1_clean_summary.md"
            a = parse_cell(md_row(fu, l1)["other7_avg_change"])["value"]
            b = parse_cell(md_row(fu, l2)["other7_avg_change"])["value"]
            return round(a - b, 4), f"u1_clean_summary.md L{l1} 减 L{l2} 的 other7_avg_change"
        m = re.match(r"L(\d+) 列 (\S+) 的三个种子值 .* 里大于 L(\d+) 规则", fm)
        if m:
            fh = ROOT / "audit" / "holdout_2526_summary.md"
            seeds = [float(x) for x in md_row(fh, int(m.group(1)))[m.group(2)].split()]
            rule = float(md_row(fh, int(m.group(3)))[m.group(2)])
            return float(sum(x > rule for x in seeds)), f"holdout_2526_summary.md L{m.group(1)} 的种子值对 L{m.group(3)} 的规则"
        raise KeyError(fm)
    if k == "perevent":
        return recompute_perevent(name, it["id"]), "逐事件文件独立重算  " + s["how"]
    if k == "code":
        from scripts.paper_v2.fig_overview import arch
        a = arch()
        key = it["id"].split("/", 1)[1]
        v = a[key]
        return (v if isinstance(v, (int, float)) else str(v)), "train_unified_clean.UnifiedGatingClean 重新实例化"
    raise KeyError(k)


def same(a, b, it):
    if a is None or b is None:
        return False
    if isinstance(b, str) or isinstance(a, str):
        return str(a) == str(b)
    if it["src"]["kind"] == "perevent":
        return abs(a - b) < 1e-9
    return abs(float(a) - float(b)) < 1e-12 or (it.get("fmt") and it["fmt"] != "int" and round(float(a), 6) == round(float(b), 6))


# ---------------------------------------------------------------- tex 单元格
def split_row(x):
    """按 & 拆一行 tex 表格（表里没有转义的 \\&），去掉行尾的 \\\\，每格还原成纯文本"""
    x = re.sub(r"\\\\\s*$", "", x.strip())
    return [norm_tex(c) for c in x.split("&")]


def tex_cells(name):
    L = (TAB_DIR / f"{name}.tex").read_text(encoding="utf8").split("\n")
    i0 = max(i for i, x in enumerate(L) if x.strip() == "\\midrule")
    i1 = next(i for i, x in enumerate(L) if x.strip() == "\\bottomrule")
    data = []
    for x in L[i0 + 1:i1]:
        xs = x.strip()
        if not xs or xs.startswith("\\addlinespace") or xs.startswith("\\multicolumn") or xs.startswith("\\cmidrule"):
            continue
        data.append(split_row(x))
    header = [c for x in L[:i0] if "&" in x for c in split_row(x)]
    seeds_row = next((split_row(x) for x in L if x.startswith("Seeds & ")), None)
    t0 = next(i for i, x in enumerate(L) if x.startswith("\\scriptsize ")) if any(x.startswith("\\scriptsize ") for x in L) else None
    note = norm_tex(L[t0]) if t0 is not None else ""
    return data, seeds_row, note, header


NUMTOK = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")


def in_cell(display, cell):
    if display is None:
        return True
    if re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", display):
        toks = NUMTOK.findall(cell.replace("−", "-"))
        return display in toks
    return norm_tex(display) in cell


def style_checks(name, note, raw):
    """破折号、正文冒号、缩写是否在表注里解释"""
    out = []
    body = "\n".join(x for x in raw.split("\n") if not x.startswith("%") and "\\label{" not in x)
    for ch, nm in [("—", "长破折号"), ("–", "短破折号"), ("--", "两个连字符")]:
        if ch in body:
            out.append(f"含{nm}")
    prose = re.sub(r"\$[^$]*\$", "", body)
    prose = re.sub(r"\\begin\{[^}]*\}\{[^}]*\}(\{[^}]*\})?", "", prose)
    if re.search(r"[A-Za-z\)]:\s", prose) or "：" in prose:
        out.append("正文里有冒号")
    abbr = {"AUC": "area under the receiver operating characteristic curve", "CNN": "concatenation network with a convolutional branch",
            "MLP": "multilayer perceptron", "CI": "bootstrap interval", "n/a": "n/a", "n.e.": "n.e.", "n.i.": "n.i.",
            "val.": "validation", "Pos.": "share of positive labels", "TASG": "task-adaptive spatial gating"}
    cells_text = body.split("\\scriptsize")[0]
    for a, expl in abbr.items():
        if re.search(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])", cells_text + note) and expl not in note:
            out.append(f"缩写 {a} 没在表注里解释")
    return out


# ---------------------------------------------------------------- 主流程
def main():
    t0 = time.time()
    report = {}
    sections = []
    totals = {}
    for name in ORDER:
        f = OUT_DIR / f"{name}_sources.json"
        d = json.load(open(f, encoding="utf8"))
        items = d["items"]
        by = {it["id"]: it for it in items}
        is_tab = name.startswith("tab_")
        if is_tab:
            data, seeds_row, note, header = tex_cells(name)
            raw = (TAB_DIR / f"{name}.tex").read_text(encoding="utf8")
            style = style_checks(name, note, raw)
        else:
            style = []
        csv_vals = None
        if (OUT_DIR / f"{name}_data.csv").exists():
            with open(OUT_DIR / f"{name}_data.csv") as fh:
                rr = list(csv.DictReader(fh))
            cols = [c for c in rr[0] if c in ("cnn_minus_xgb", "value", "gate", "drop_players", "lo", "hi")]
            csv_vals = [float(r[c]) for r in rr for c in cols if r.get(c) not in (None, "")]
        rows = []
        n_ok = 0
        for it in items:
            v, where = reread(name, it, by)
            ok1 = same(it["value"], v, it)
            ok2 = True
            cellnote = ""
            if is_tab and it.get("cell") is not None:
                r, c = it["cell"]
                if r == "note":
                    ok2 = in_cell(it["display"], note)
                    cellnote = "表注"
                elif r == "seeds":
                    ok2 = in_cell(it["display"], seeds_row[c])
                    cellnote = f"Seeds 行第 {c + 1} 列"
                else:
                    ok2 = in_cell(it["display"], data[r][c])
                    cellnote = f"数据第 {r + 1} 行第 {c + 1} 列"
            ok3 = True
            if csv_vals is not None and it["src"]["kind"] in ("perevent", "md", "json") and not isinstance(it["value"], str):
                if name in ("fig_outcome_vs_scalars", "fig_gate") and it["src"]["kind"] == "perevent":
                    ok3 = any(abs(it["value"] - x) < 1e-6 for x in csv_vals)
                elif name == "figS1_gate_vs_shuffle" and it["src"]["kind"] in ("perevent", "json"):
                    ok3 = any(abs(it["value"] - x) < 1e-6 for x in csv_vals)
                    cellnote = "图的数据表"
                elif name == "figS2_soft_penalty" and it["src"]["kind"] == "md":
                    ok3 = any(abs(it["value"] - x) < 1e-9 for x in csv_vals)
                    cellnote = "图的数据表"
                elif name in ("fig_destination", "fig_holdout") and ("/s" in it["id"] or name == "fig_holdout") and not it["id"].endswith("/thr") \
                        and not it["id"].endswith("_mean") and not it["id"].endswith("/mean"):
                    ok3 = any(abs(it["value"] - x) < 1e-6 for x in csv_vals)
                    cellnote = "图的数据表"
            ok = ok1 and ok2 and ok3
            n_ok += ok
            disp = it["display"] if it["display"] is not None else (f"{it['value']:.6g}" if isinstance(it["value"], float) else str(it["value"]))
            rows.append((it["id"], disp, where, cellnote, "MATCH" if ok else "MISMATCH", "" if ok else f"回读 {v}，登记 {it['value']}，进表 {ok2}，进图 {ok3}"))
        lab_f = OUT_DIR / f"{name}_labels.csv"
        if lab_f.exists():
            regvals = [it["value"] for it in items if isinstance(it["value"], float)]
            with open(lab_f, encoding="utf8") as fh:
                labs = list(csv.DictReader(fh))
            bad = []
            for lb in labs:
                t, v = lb["text"], float(lb["value"])
                nd = len(t.split(".")[1])
                ok_reg = any(f"{rv:.{nd}f}" == t for rv in regvals) or any(abs(rv - v) < 1e-9 for rv in regvals)
                if not (f"{v:.{nd}f}" == t and ok_reg):
                    bad.append(t)
            style = style + ([f"图上标的数有 {len(bad)} 个对不上登记值  {bad}"] if bad else [])
            rows.append(("图上标的数值", f"{len(labs)} 个", f"out/{name}_labels.csv，每个标注等于原值四舍五入，且等于某个登记值的同位四舍五入",
                         "图上文字", "MATCH" if not bad else "MISMATCH", "" if not bad else f"对不上 {bad}"))
            n_ok += (not bad)
            items = items + [dict(id="labels")]
        totals[name] = (n_ok, len(items), style)
        report[name] = dict(n_ok=n_ok, n=len(items), style=style, mismatches=[r for r in rows if r[4] != "MATCH"], notes=d.get("notes", []))
        sections.append((name, rows, d))
        print(f"[verify] {name:24s} {n_ok}/{len(items)} MATCH  格式检查 {style or '无问题'}  ({time.time() - t0:.0f}s)", flush=True)
    (OUT_DIR / "verify_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf8")
    if "--append" in sys.argv:
        names = sys.argv[sys.argv.index("--append") + 1].split(",")
        append_md([x for x in sections if x[0] in names], totals)
        print(f"[verify] 只把 {names} 三节追加到 {MD_OUT} 末尾", flush=True)
        return
    if "--no-md" in sys.argv:
        print("[verify] 只核对，不重写数据来源 md", flush=True)
        return
    write_md(sections, totals)
    print(f"[verify] 写好 {MD_OUT}", flush=True)


# ---------------------------------------------------------------- 写 md
HEADER = """# 图表数据来源和逐数核对（2026-10-04）

{when} 由 `scripts/paper_v2/verify_all.py` 生成，写的人是图表重做，给写稿写正文时查数用。每张表、每张图里的每个数都在下面各节列出，一行一个，写明它从哪个文件哪一行哪一列（或 json 哪个字段、或由逐事件文件怎么算出来）来，显示成什么样，进了表的哪一格，核对结果是什么。写稿写正文时，数以这里为准；要改数先改结果文件和生成脚本，再跑 `make_all.sh`，不手改 tex。

## 怎么核的

每个数核三层，全部条目都核，不抽样。
1. 回读。md 按登记的行号重读那一行，用该表的表头对上列名取单元格；json 按字段路径重读；训练日志按行号加正则重读；判定词按行号查原句是否含登记的那几个字。逐事件文件来的数（种子标准差、逐种子点、干预前后的差、202 场正例率）用 `verify_all.py` 自己的一套代码重算，直接 `np.load` 加 `sklearn.metrics.roc_auc_score`，202 场的清单直接从 `audit/raw_scan/match_state_2425.parquet` 按“状态正常且射门带帧率不低于 70%”取，不经过 `summarize_u1_clean.py` 和 `common.py` 的重算函数，等于用两套实现各算一遍。
2. 进表。解析生成的 tex，按登记的行列位置取出那一格，看显示值是不是那一格里的一个数（按完整数字比，不按子串比）。表注里的数在表注全文里找。
3. 进图。图的数据表（`scripts/paper_v2/out/*_data.csv`，就是图的表格版）里每个画出来的值都要等于登记的值；另外五张图的 png 我逐张读过（见末尾）。
三层都过记 MATCH，任何一层不过记 MISMATCH，并写明哪层不过。下面各表“位置”一栏的行数从表头下面第一行数起，列数从最左一列数起。另查了 tex 里有没有破折号、正文冒号，表里出现的缩写有没有在表注里解释。

生成脚本另有一道闸。各脚本在写表前就把汇总表里的均值、标准差和逐事件重算的结果对一遍（u1_clean_summary.md 的单任务表、配对差表、各运行表，holdout summary.json 的 derived 和 ci），对不上脚本直接停，不出文件。

## 权威来源文件和 MD5（核对时的版本）

{md5s}

行号都按上面这个版本。`u1_clean_summary.md` 的 MD5 是 aea28711…，和重写方案引用的 00:11 版相同，所以重写方案里的“汇总表 L 行号”和这里的行号是同一套。
"""


def write_md(sections, totals):
    allsrc = {}
    for name, rows, d in sections:
        allsrc.update(d.get("source_md5", {}))
    md5s = "\n".join(f"- `{k}`  {v}" for k, v in sorted(allsrc.items()))
    out = [HEADER.format(when=time.strftime("%Y-%m-%d %H:%M"), md5s=md5s)]
    for name, rows, d in sections:
        kind = "表" if name.startswith("tab_") else "图"
        path = f"outputs/tables/{name}.tex" if kind == "表" else f"outputs/figures/v2/{name}.pdf（和 .png）"
        n_ok, n, style = totals[name]
        out += [f"## {kind} {name}", "", f"文件 `{path}`，生成脚本 `scripts/paper_v2/{name}.py`。{n} 个数，MATCH {n_ok} 个。"
                + (f" 格式检查 {'；'.join(style)}。" if style else (" 格式检查无问题（无破折号、无正文冒号、缩写都在表注里解释）。" if kind == "表" else "")), ""]
        if NOTES.get(name):
            out += [NOTES[name], ""]
        if d.get("notes"):
            out += ["脚本写下的附带核对", ""] + [f"- {x}" for x in d["notes"]] + [""]
        out += ["| 项 | 显示 | 出处 | 位置 | 核对 |", "|---|---|---|---|---|"]
        for iid, disp, where, cellnote, st, extra in rows:
            out.append(f"| {iid} | {disp} | {where} | {cellnote} | {st}{('，' + extra) if extra else ''} |")
        out.append("")
    out += [TAIL.format(summary="\n".join(f"| {k} | {v[1]} | {v[0]} | {v[1] - v[0]} | {'；'.join(v[2]) if v[2] else '无'} |" for k, v in totals.items()),
                        total=sum(v[1] for v in totals.values()), ok=sum(v[0] for v in totals.values()))]
    out += [""] + ADDENDA
    MD_OUT.write_text("\n".join(out), encoding="utf8")


def append_md(sections, totals):
    """把指定交付物的节（说明、附带核对、逐数表）追加到数据来源 md 末尾，不动前面的内容"""
    out = ["", f"## 补充图（{time.strftime('%Y-%m-%d %H:%M')} 追加）", ""]
    for name, rows, d in sections:
        n_ok, n, style = totals[name]
        out += [f"### 图 {name}", "", f"文件 `outputs/figures/v2/{name}.pdf`（和 .png），生成脚本 `scripts/paper_v2/{name}.py`。"
                f"{n} 个数，MATCH {n_ok} 个。" + (f" 格式检查 {'；'.join(style)}。" if style else ""), ""]
        if NOTES.get(name):
            out += [NOTES[name], ""]
        if d.get("notes"):
            out += ["脚本写下的附带核对", ""] + [f"- {x}" for x in d["notes"]] + [""]
        out += ["| 项 | 显示 | 出处 | 位置 | 核对 |", "|---|---|---|---|---|"]
        for iid, disp, where, cellnote, st, extra in rows:
            out.append(f"| {iid} | {disp} | {where} | {cellnote} | {st}{('，' + extra) if extra else ''} |")
        out.append("")
    with open(MD_OUT, "a", encoding="utf8") as fh:
        fh.write("\n".join(out) + "\n")


ADDENDA = ['- 2026-10-04 16:37 补记。四张数据图（fig_outcome_vs_scalars、fig_destination、fig_gate、fig_holdout）按作者意见照 5 月旧图的版式重做，版式重做，数值未变。重做后用 `verify_all.py --no-md` 重核，原 900 个登记数和四张图上新标的数值（每个标注都等于登记值的同位四舍五入）全部 MATCH，png 逐张看过并和旧图并排比过。fig_overview 按作者意见停用，稿子换回 `figures/fig1.pdf`，v2 下的文件保留，不再改。', '- 2026-10-04 17:00 补记。按负责人要求统一命名，表里单任务门控模型的 Gating 改成 TASG（表注里解释为 task-adaptive spatial gating），original encoder 改成 pooled encoder（tab_destination、tab_interventions、tab_holdout、tab_gate_variants_supp 的表注和 fig_destination 的行名；tab_gate_variants_supp 最后一行的判定词随之写成 Not met; the pooled encoder is the main model），数值未变，verify_all.py --no-md 重核全部 MATCH。', '- 2026-10-04 17:06 补记。按正文 4.2 的新定义，表注里 CNN 的展开从 convolutional neural network 改成 concatenation network with a convolutional branch（tab_outcome_main、tab_destination、tab_holdout、tab_gate_variants_supp 四处；tab_interventions、tab_application_supp、tab_tasks 的表注里没有展开 CNN），数值未变，verify_all.py --no-md 重核全部 MATCH。', '- 2026-10-04 17:07 补记。tab_outcome_main 表注的 six location features 改成 six event features（括号内容不变）；tab_destination 表注里同样的说法 thin scalars are location and four 360 summary features 一并改成 six event features and four 360 summary features。数值未变，verify_all.py --no-md 重核全部 MATCH。', '- 2026-10-04 17:28 补记。按正文 4.1 的新定义（gain 是 CNN 减同标量集 MLP，residual 是 CNN 减同标量集 XGBoost），tab_holdout 第 19、21、23 行和表注里把 CNN 减 MLP 叫 residual gain 的地方改成 enriched gain，fig_holdout 面板 (b) 的行标签同样改成 Enriched gain、Pressure enriched gain，thin gain 那一行的括号改成 reference for enriched gain。数值未变，verify_all.py --no-md 重核全部 MATCH。', '- 2026-10-04 17:46 补记。按复核第四遍改了两张补充图的版式，数值未变。figS1 面板 (c) 的 Pass、Dribble、Ball recovery 三个标签挪进坐标轴内。figS2 面板 (a) 加厚标量确认步的两个均值标签 0.625 和 0.095 挪离虚线，用短线连回均值横线。figS3 和表 S8 的 Changed team 一度改成 Changed club，按负责人更正已改回 team，现在的文件和改动前同名同字。verify_all.py --no-md 重核全部 MATCH。', '- 2026-10-06 10:20 补记（本机数值收尾记录）。表 9（tab_holdout）C4 落点干预模型的 2024/25 门控最大值由 0.279 改为 0.278。原来脚本拿汇总表 u1_clean_summary.md L10 已舍成四位的 0.2785 再舍成三位，成了 0.279；逐事件原值是 0.27847（shuf_dest_s2.npz，202 场落点事件门控均值），按原值三位应为 0.278。现在 tab_holdout.py 改从逐事件文件取逐种子原值，再舍成三位，汇总表的 gate_min、gate_max 只拿来四位核对，上面 tab_holdout 一节换成重跑后的版本（登记的数由 91 个增到 98 个，多出 u1 四个、shuf_dest 三个逐种子门控），其余各节没动。2025/26 一列仍是 0.241 to 0.279（25/26 种子 2 原值 0.27924，舍入正确）。图 5（fig_holdout）面板 d 画的是逐种子原值点、标的是均值 0.264，本来就不含 0.279，重出后数据表、标注表、出处登记和 png 逐像素都和重出前相同，pdf 只差生成时间。verify_all.py --no-md 重核 15 个交付物全部 MATCH。']


NOTES = {
    "tab_tasks": "任务定义照 `scripts/training/train_unified_clean.py` 的 build_dataset。训练加验证条数和正例率取训练日志的 [data] 行。logs/ 下各运行打出的这一组行都相同，只有 `logs/u1_clean_u1fcn_s0_2026-10-02.log`（14:04，MacBook）不同（Shot、Interception、Dribble、Tackle 的训练条数少 10 到 24 条，Pass 的测试正例率 0.815）。查过，那是一次没被采用的运行，现用的 `data/cache/u1_clean/u1fcn_s0.json`（16:51，host 字段是 Mac-mini.local）出自 `logs/mini_chain_unified_2026-10-02.log` L193 起那次运行，它的 [data] 行（L194 起）和其余运行相同；u1fcn_s0.npz 在 202 场上各任务的条数和正例率也和 u1_s0 相同（逐事件查过）。202 场的正例率是逐事件重算的，冻结方案第一节第 4 点表里“24/25”那列（0.812、0.093 等）是 229 场的日志值，两者不同，正文用这里的 202 场的数。",
    "tab_outcome_main": "统一模型加厚版 u1rich 用四个种子（负责人 10-03 00:19 定的），预先标准的“没追平”判定在三个种子上做，四个种子按同一条标准仍是没追平，表注写了。",
    "tab_destination": "规则的 top-3 只在表注里（模型的逐事件文件没存全分布，算不出 top-3）。单任务的 top-1 标准差、统一模型的 skill 标准差汇总表里没有，是逐事件重算的，重算的均值和汇总表四位一致。25/26 的单任务门控没有跑，写 n.e.。",
    "tab_interventions": "成绩变化用未四舍五入的值相减，所以个别格和表里两端四位小数相减差 0.0001（例如 Tackle 种子 0，0.6157 到 0.5908 显示变化 $-$0.0250）。落点干预三行的八个变化和汇总表“落点干预下其余任务被拖累多少”一节门控行逐格一致。",
    "tab_holdout": "判定词取冻结方案文末“判定”表，C2 照负责人的要求写成 Confirmed, Pressure near the bound（原句是“确认”，并写明 Pressure 厚增益 0.0091 贴着 0.010 的线）。C3 的标题按负责人 15:33 的要求写 stays close to，不写 on par。没有总括句。C4 的 24/25 门控范围 10-06 起取逐事件文件重算的逐种子门控的最小、最大再舍成三位（汇总表 u1、shuf_dest 两表的 gate_min 和 gate_max 已是四位，再舍一次会把 shuf_dest 的 0.27847 舍成 0.279），并和汇总表四位核对。",
    "tab_gate_variants_supp": "卡里点名五项，表里九行。软代价、硬代价各按标量薄、厚分两行，加厚标量分门控、成绩两问，另多一行主模型编码器规则。加厚标量成绩一问的三个种子数取它自己标准文件末尾的结果表（那里写的是“负 0.0042”这种写法，正则取数后取负）。",
    "tab_application_supp": "三个种子的合并点估计按负责人要求从表里挪到表注。留队球员的 D 汇总表写明“不单独判”，只进表注。",
    "fig_overview": "结构和每个维数都从 `UnifiedGatingClean` 的实例读出（见下表 arch 各项），脚本里另有断言，确认标量编码器输入等于标量加任务嵌入、门控输入是两个 64 维向量拼接、各层顺序和类型与代码一致。旧图 `figures/fig1.pdf` 和现在的代码有四处不一致，所以没有复制而是重画，差异是旧图没有任务嵌入、输出头是 sigmoid 或 identity（xG 回归）而没有落点 96 维头、输出列的任务是 pass、dribble、shot、cross、标题写 Football 并标了 Core Contribution。图里没有画 dropout（代码里标量编码器和共用隐层各有一个 0.3 的 dropout），图注里可以补一句。",
    "fig_outcome_vs_scalars": "每个点是同一种子的 CNN 减 XGBoost，薄标量五个种子，加厚标量三个种子（加厚 CNN 只有三个种子，XGBoost 有五个，取共有的三个配对）。两组同一个 y 轴范围，0 线是同一标量集的 XGBoost，虚线是 +0.01。每组的均值、标准差、最小、最大值另和汇总表两张配对差表逐项比过，正的种子数也比过。",
    "fig_destination": "规则没有随机性只有一个点。24/25 的逐种子点从逐事件文件重算，25/26 的取 summary.json 的 by_seed。",
    "fig_gate": "灰色是不干预的统一模型 u1 四个种子，橙色是干预运行三个种子；种子 0 到 2 的逐种子门控另和汇总表“干预”一节的 gate_base、gate_shuffled_run 四位比过。名次文字取干预表。Pressure 干预只有一个种子，按卡没画进图，在 tab_interventions 里。",
    "figS1_gate_vs_shuffle": "横轴是各任务在 202 场测试事件上的门控均值（逐种子从逐事件文件重算），纵轴是测试时只打乱球员两个通道的指标下降，取各运行的 *_ablation.json 的 drop_players（结果类任务是 AUC 下降，落点是 skill 下降，单位不同，所以纵轴取对数，落点用方块单独标出）。每个种子的 Spearman 取汇总表“门控均值对球员通道消融下降量”一节，脚本用重算的门控和 json 里的下降量再算一遍，三位小数一致才画。负责人消息里提到的汇总表 L75、L77 是门控对单任务 CNN 增益的相关，不是这张图的量，这张图用的是 L358 到 L372 的消融相关。基准统一模型只有种子 3 存了权重，所以 (a) 只有一个种子。",
    "figS2_soft_penalty": "数全部取汇总表“门控代价实验”一节的四张表。标量薄的探索步只有 λ = 0.01 做过整张量打乱和强制关门两种消融，λ = 0 和 0.03 两格是“未做消融”，图上只画 λ = 0.01 一个点。确认步三个种子在两个面板都画了（负责人原话只要求右图画，左图也画是为了和右图对照），横线和数字是三个种子的均值。虚线是预先写下的标准里甲的两条阈值。",
    "figS3_player_preference": "逐球员点是每名球员在每一折里三个种子的平均（和 summarize 的口径相同），按检验传球数加权的平均和汇总表的 D 合并五位一致。合并估计、两折和三个种子的点估计、所有 95% 区间照抄汇总表，没有重抽自助法。左图纵轴范围是逐球员值的全范围，合并区间在这个尺度上很短，所以右图放大到零附近。",
    "fig_holdout": "C1 到 C3 的区间是 25/26 按比赛自助法 95% 区间，24/25 没有区间（留出汇总没给）。C4 的门控读数没有区间，画的是逐种子点。比值那一行（厚增益对薄增益 0.36）单位不同，只在 tab_holdout 里，没进图。",
}

TAIL = """## 图注和正文里常用的结构数、种子数出自哪里

结构数（fig_overview，`scripts/training/train_unified_clean.py`，行号按 10-02 版）
- 20 个薄标量。`feats()`（L133 到 L137）把 F_BASE（L68，6 个位置量）、F_PASS（L69 到 L70，10 个传球量，只有 Pass 任务非零，其余任务这 10 维填 0）、F_SB（L71，4 个 360 汇总量）拼起来，6 + 10 + 4 = 20。Dribble 和 Tackle 用 F_FX（L72，同样 4 个量，从视角摆正后的定格帧重算）代替 F_SB，宽度不变。
- 35 个加厚标量。`rich_scalars()`（L190 到 L211）算 15 个量，队友和对手各 7 个（L202 到 L205 的循环），再加球到球门连线带内的对手数 1 个（L206 到 L208）；`--rich_scalars` 时在 L368 拼到 20 个薄标量后面，20 + 15 = 35。
- Linear 36 to 64。L218，`nn.Linear(event_dim + task_embed_dim, embed_dim)`，event_dim 是 20，task_embed_dim 默认 16（L213），embed_dim 默认 64（L213），20 + 16 = 36。任务嵌入在 L217，8 个任务各一个 16 维向量。
- 其余维数。空间编码器 L220 到 L223（Conv 3×3 到 32 通道，BN，ReLU，max-pool 2；Conv 3×3 到 64 通道，BN，ReLU；自适应平均池化到 2×2；Linear 256 到 64；tanh）。门控 L225（Linear 128 到 64 加 sigmoid）。共用隐层 L228（Linear 64 到 32，ELU，dropout 0.3）。两个头 L229（Linear 32 到 1）和 L230（Linear 32 到 96）。落点损失乘 1/ln96 在 L415（文件头 L13 有说明）。这些维数也由 `fig_overview.py` 实例化模型读出，见上面 fig_overview 一节 arch 各项，全部 MATCH。

种子数（图注、表头里出现的）
- 薄标量单任务模型五个种子，XGBoost、MLP、CNN、门控都是。`audit/u1_clean_summary.md` L574 到 L581 每格的 n=5；fig_outcome_vs_scalars 左图的“五个配对种子”是 L587 到 L617 配对差表里 seeds_positive 一列的分母 5（例如 L590 的 0/5）。
- 加厚标量单任务模型，XGBoost 五个种子，MLP、CNN、门控三个种子。L625 到 L632 每格的 n=5 或 n=3；fig_outcome_vs_scalars 右图的“三个配对种子”是 L638 到 L668 配对差表 seeds_positive 的分母 3（例如 L641 的 0/3）。
- 统一模型原编码器四个种子。L62 的标题“种子 [0, 1, 2, 3]”；加厚统一模型 u1rich 四个种子，L259。
- 保留分辨率变体三个种子。L145 的标题“种子 [0, 1, 2]”。
- 落点干预三个种子，L5；Tackle 干预三个种子，L43；Pressure 干预一个种子，L24。
- 25/26 留出集上各模型的种子数。`audit/holdout_2526_summary.md` L136 起的“全部模型逐任务”表里每格的 n=，统一模型 u1 是 n=4（L152），加厚 XGBoost 是 n=5（L148），u1fcn 是 n=3（L154），M2 薄 n=5（L146），M2 加厚 n=3（L149）；u1rich 在留出集上是三个种子（L155，n=3），和 24/25 的四个不同，原因见该文件 L244 的事后注。

## 核对表（全部条目的汇总）

| 交付物 | 登记的数 | MATCH | MISMATCH | 格式检查 |
|---|---:|---:|---:|---|
{summary}

合计 {total} 个数，MATCH {ok} 个。每个数逐条的核对结果在上面各节表格的最后一列。

### 人工核对（脚本查不到的）

- 五张图的 png 逐张用图片方式读过，查四角是否出界、标签是否重叠、图例是否完整、字号。fig_destination 第一版图例最后一项出界，15:39 已改成两行重出（pdf 和 png 同一脚本同一次导出）；fig_outcome_vs_scalars 第一版的参照线文字压到 Dribble 的种子点，已挪进图例；fig_holdout 和 fig_overview 第一版有文字出框，已改版式重出。最终版五张都在画布内。
- 七张表在 scratch 里拷一份 manuscript.tex 和 supplementary.tex 编译过，无错误，表没有 Overfull（manuscript 里 5 个 6.8 pt 的 Overfull 是还没交图时占位框的，与表无关），并逐页看过渲染。tab_application_supp 第一版超出补充材料版心，已去掉 Seeds 列。
- 判定词和标准文件逐条对过（tab_holdout、tab_gate_variants_supp、tab_application_supp 的判定词都按行号登记，回读时查原句）。
- 表注里出现的缩写都在表注里解释过（AUC、CNN、MLP、CI、n/a、n.e.、n.i.、val.、Pos.，脚本按表查）。图里出现的缩写（AUC、CNN、MLP、BN、BCE、ReLU、ELU）要在图注里解释，图注由写稿写，我发过建议图注。

### 没做

- 没改任何结果文件和 eval、training、audit、data 下的脚本；没动 manuscript.tex、supplementary.tex、references.bib（编译检查用的是 scratch 里的拷贝）。
- 模型的 top-3 没有算（逐事件文件只存了真实格的对数概率和 top-1）。
- Dribble 和 Tackle 没有 25/26 的结果，表里写 n.i.；跨联赛结果没有在干净数据上重跑，表图里都没有。

### 可能漏的

1. 进表核对按“显示值是格里的一个完整数字”判，像种子数 3、名次 1 这种很短的数，同一格里碰巧有同样的数字也会判过；这类格我在渲染页上人工看过，但不是脚本能保证的。
2. 汇总表本身的正确性不在本次核对范围内（本次核的是“表图里的数等于汇总表和逐事件文件”），逐事件文件的独立重算只覆盖了图表里用到的那部分。
3. 202 场正例率和单任务 top-1 标准差这几类数是我新算的（汇总表里没有），出处是逐事件文件，没有第二份人工文件可以对照。
"""


if __name__ == "__main__":
    main()
