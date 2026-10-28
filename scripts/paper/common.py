#!/usr/bin/env python3
"""
common.py, shared tools of the table and figure scripts
=======================================================
Every tab_*.py and fig*.py script reads its numbers through this module. Each number that enters a table or figure is
registered with its source (file, line and column, json key path, or the registered numbers it is computed from), and
the registry is written to outputs/provenance/{name}_sources.json.
  markdown table  reads one row by line number, matches the cells to the column names of the table header, and can split
                  a cell of the form "mean ± sd (value) n=seeds" into its parts
  json            reads a value by key path
  per event       the per-seed values of 2024/25 are recomputed from the per-event predictions in data/cache, with the
                  functions of scripts/eval/summarize_u1_clean.py
  decision rules  rule_line finds the row of a recorded value or verdict in results/decision_rules.md
This module only reads result files and writes nothing under data/ or scripts/.

Inputs   results/*.md and *.json, data/cache/u1_clean/*.npz, data/cache/single_clean{,_rich}/preds/*.npz
Outputs  none (imported by the other scripts)

Usage
  PYTHONPATH=. python -c "import scripts.paper.common"   (from the repository root)

Last modified 2026-10-28
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

DEV = Path(__file__).resolve().parents[2]          # repository root
ROOT = DEV                                          # provenance entries are written relative to this
MANU = ROOT / "outputs"                             # generated tables and figures
TAB_DIR = MANU / "tables"
FIG_DIR = MANU / "figures"
OUT_DIR = DEV / "results" / "figure_data"           # plotted values of every figure (*_data.csv, *_labels.csv)
PROV_DIR = MANU / "provenance"                      # provenance registry of every table and figure (*_sources.json)

SRC = {
    "u1": DEV / "results" / "u1_clean_summary.md",
    "dest_rules": DEV / "results" / "dest_rule_baselines.md",
    "pref": DEV / "results" / "player_decision_preference_summary.md",
    "hold_md": DEV / "results" / "holdout_2526_summary.md",
    "hold_json": DEV / "results" / "holdout_2526_summary.json",
    "hold_dest_rules": DEV / "results" / "holdout_2526_dest_rules.json",
    "train_log": DEV / "logs" / "u1_s0_training.log",
    "decision_rules": DEV / "results" / "decision_rules.md",
}

TASKS = ["pass", "dest", "shot", "interception", "ball_recovery", "pressure", "dribble", "tackle"]
OUTCOME = [t for t in TASKS if t != "dest"]
TASK_NAME = {"pass": "Pass", "dest": "Destination", "shot": "Shot", "interception": "Interception",
             "ball_recovery": "Ball recovery", "pressure": "Pressure", "dribble": "Dribble", "tackle": "Tackle"}


def rel(p: Path) -> str:
    """Path relative to the repository root, as written in the provenance entries (symbolic links are not resolved)"""
    return str(Path(p).absolute().relative_to(ROOT))


def md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- markdown tables

def _lines(path):
    return Path(path).read_text(encoding="utf8").split("\n")


def _split(line):
    parts = line.strip().split("|")
    return [c.strip() for c in parts[1:-1]]


def md_row(path, line):
    """Row at line number line (from 1) of a markdown table, as {column name: cell string}"""
    L = _lines(path)
    i = line - 1
    assert L[i].strip().startswith("|"), f"{rel(path)} L{line} is not a table row  {L[i][:60]}"
    j = i
    while j > 0 and not re.match(r"^\|\s*:?-{3}", L[j].strip()):
        j -= 1
    header = _split(L[j - 1])
    cells = _split(L[i])
    assert len(header) == len(cells), f"{rel(path)} L{line} has a different number of cells than its header"
    return dict(zip(header, cells))


def md_find(path, heading_prefix, nth_table=0, **keys):
    """Line number (from 1) of the row whose columns equal keys, in table nth_table after the heading that starts with heading_prefix"""
    L = _lines(path)
    start = next(i for i, x in enumerate(L) if x.startswith(heading_prefix))
    k, i = -1, start + 1
    while i < len(L):
        if L[i].startswith("#") and i > start + 1 and k >= nth_table:
            break
        if re.match(r"^\|\s*:?-{3}", L[i].strip()):
            k += 1
            if k == nth_table:
                header = _split(L[i - 1])
                j = i + 1
                while j < len(L) and L[j].strip().startswith("|"):
                    row = dict(zip(header, _split(L[j])))
                    if all(row.get(c) == str(v) for c, v in keys.items()):
                        return j + 1
                    j += 1
                raise KeyError(f"{rel(path)} {heading_prefix} table {nth_table} has no row {keys}")
        i += 1
    raise KeyError(f"{rel(path)} after {heading_prefix} there is no table number {nth_table}")


def rule_line(key, path=None):
    """Line number (from 1) of the table row in results/decision_rules.md whose first cell equals key"""
    L = _lines(SRC["decision_rules"] if path is None else path)
    hits = [i + 1 for i, x in enumerate(L) if x.strip().startswith("|") and _split(x)[:1] == [key]]
    assert len(hits) == 1, f"{key!r} found in {len(hits)} table rows of the decision rules"
    return hits[0]


def parse_cell(s):
    """Splits a cell into parts. '0.2710 ± 0.0004 (0.0947) n=5' gives mean, sd, paren and n; a plain number gives value;
    a bracketed interval gives lo and hi"""
    s = s.strip().replace("−", "-")
    out = {}
    m = re.search(r"n=(\d+)", s)
    if m:
        out["n"] = int(m.group(1))
    m = re.search(r"\(([-+0-9.]+)\)", s)
    if m:
        out["paren"] = float(m.group(1))
    m = re.match(r"^([-+]?[0-9.]+)\s*±\s*([0-9.]+|nan)", s)
    if m:
        out["mean"] = float(m.group(1))
        out["sd"] = float(m.group(2))
    m = re.match(r"^\[([-+0-9.]+),\s*([-+0-9.]+)\]$", s)
    if m:
        out["lo"], out["hi"] = float(m.group(1)), float(m.group(2))
    if not out:
        try:
            out["value"] = float(s.replace("%", "").replace(",", ""))
        except ValueError:
            out["text"] = s
    return out


def md_value(path, line, col, part="value"):
    cell = md_row(path, line)[col]
    p = parse_cell(cell)
    if part not in p:
        raise KeyError(f"{rel(path)} L{line} column {col} cell {cell!r} has no {part}")
    return p[part]


def json_value(path, keys):
    d = json.load(open(path, encoding="utf8"))
    for k in keys:
        d = d[k]
    return d


# ---------------------------------------------------------------- provenance registry

class Registry:
    """One entry per number in a table or figure. src locates the number so that it can be read again; display is the
    string shown in the table or figure"""

    def __init__(self, name):
        self.name = name
        self.items = []
        self.notes = []

    def add(self, iid, value, src, display=None, cell=None, fmt=None):
        it = dict(id=iid, value=None if value is None else float(value) if isinstance(value, (int, float, np.floating, np.integer)) else value,
                  src=src, display=display, cell=cell, fmt=fmt)
        self.items.append(it)
        return value

    # common kinds of source
    def md(self, iid, path, line, col, part="value", fmt=None, cell=None):
        v = md_value(path, line, col, part)
        d = fmt_num(v, fmt) if fmt else None
        self.add(iid, v, dict(kind="md", file=rel(path), line=line, col=col, part=part), display=d, cell=cell, fmt=fmt)
        return v

    def mdre(self, iid, path, line, col, regex, group=1, fmt=None, cell=None, sign=1):
        """Number taken from a cell by a regular expression; with col None the whole line is searched. sign=-1 negates the match"""
        s = md_row(path, line)[col] if col else _lines(path)[line - 1]
        m = re.search(regex, s.replace("−", "-"))
        assert m, f"{rel(path)} L{line} column {col} does not match {regex!r}  {s[:80]}"
        v = sign * float(m.group(group))
        d = fmt_num(v, fmt) if fmt else None
        self.add(iid, v, dict(kind="mdre", file=rel(path), line=line, col=col, regex=regex, group=group, sign=sign), display=d, cell=cell, fmt=fmt)
        return v

    def js(self, iid, path, keys, fmt=None, cell=None):
        v = json_value(path, keys)
        d = fmt_num(v, fmt) if fmt else None
        self.add(iid, v, dict(kind="json", file=rel(path), path=list(keys)), display=d, cell=cell, fmt=fmt)
        return v

    def derived(self, iid, value, formula, inputs, fmt=None, cell=None):
        d = fmt_num(value, fmt) if fmt else None
        self.add(iid, value, dict(kind="derived", formula=formula, inputs=inputs), display=d, cell=cell, fmt=fmt)
        return value

    def perevent(self, iid, value, how, fmt=None, cell=None):
        d = fmt_num(value, fmt) if fmt else None
        self.add(iid, value, dict(kind="perevent", how=how), display=d, cell=cell, fmt=fmt)
        return value

    def text(self, iid, text, path, line, must_contain, cell=None):
        L = _lines(path)
        assert must_contain in L[line - 1], f"{rel(path)} L{line} does not contain {must_contain!r}"
        self.add(iid, text, dict(kind="text", file=rel(path), line=line, must_contain=must_contain), display=text, cell=cell)
        return text

    def save(self, extra=None):
        PROV_DIR.mkdir(parents=True, exist_ok=True)
        srcs = sorted({it["src"]["file"] for it in self.items if "file" in it["src"]})
        d = dict(name=self.name, items=self.items, notes=self.notes,
                 source_md5={f: md5(ROOT / f) for f in srcs})
        if extra:
            d.update({k: (rel(v) if k == "data_csv" else v) for k, v in extra.items()})
        (PROV_DIR / f"{self.name}_sources.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf8")


def fmt_num(v, fmt):
    """fmt is 'f4' (four decimals), 'f3', 'f5', 'pct1' (percent with one decimal), 'int' (integer with thousands separator), 'sf4' (four decimals with sign), 'sf5'"""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    if fmt.startswith("sf"):
        nd = int(fmt[2:])
        s = f"{v:+.{nd}f}"
        return s.replace("+0." + "0" * nd, "0." + "0" * nd).replace("-0." + "0" * nd, "0." + "0" * nd)
    if fmt.startswith("f"):
        s = f"{v:.{int(fmt[1:])}f}"
        return "0." + "0" * int(fmt[1:]) if s == "-0." + "0" * int(fmt[1:]) else s
    if fmt == "pct1":
        return f"{100 * v:.1f}"
    if fmt == "int":
        return f"{int(round(v)):,}"
    raise ValueError(fmt)


def tex(s):
    """Converts a display string to LaTeX, with a math minus sign"""
    return re.sub(r"(?<![0-9A-Za-z])-(?=[0-9.])", "$-$", s).replace("+0.", "$+$0.").replace("±", "$\\pm$")


def norm_tex(s):
    """Converts a LaTeX cell back to plain text, for comparison with the registered display strings"""
    s = s.replace("$-$", "-").replace("$+$", "+").replace("$\\pm$", "±").replace("$\\times$", "×")
    s = re.sub(r"\\makecell(\[[a-z]\])?", "", s)
    s = re.sub(r"\\(scriptsize|footnotesize|small|textsc|textit|textbf|emph)", "", s)
    s = s.replace("\\\\[-2pt]", " ").replace("\\\\", " ").replace("{", "").replace("}", "").replace("$", "").replace("\\,", "").replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- per-seed values of 2024/25 (recomputed from the per-event files)

_RUNS = None


def unified_runs():
    """All unified model runs recomputed on the 202 matches with summarize_u1_clean.load_runs, as {(tag, seed): run}"""
    global _RUNS
    if _RUNS is None:
        import os
        cwd = os.getcwd()
        os.chdir(DEV)
        try:
            from scripts.eval.summarize_u1_clean import load_runs
            _RUNS = {(r["tag"], r["seed"]): r for r in load_runs()}
        finally:
            os.chdir(cwd)
    return _RUNS


def unified_seed_values(tag, task, what):
    """what is 'metric' (top-1 for dest, AUC otherwise), 'skill' or 'gate'. Returns {seed: value}"""
    out = {}
    for (tg, sd), r in sorted(unified_runs().items()):
        if tg != tag:
            continue
        t = r["tasks"][task]
        if what == "metric":
            out[sd] = t["test"]["top1"] if task == "dest" else t["test"]["auc"]
        elif what == "skill":
            out[sd] = t["test"]["skill"]
        elif what == "gate":
            out[sd] = t["gate_mean"]
    return out


_SINGLE = {}


def single_seed_values(dirname, task, model, what="metric"):
    """Per-seed values of a single-task model on the 202 matches, computed as in summarize_u1_clean.single_task_table.
    what is 'metric' (AUC, top-1 for dest) or 'skill' (for dest, 1 minus log loss over ln96). Returns {seed: value}"""
    key = (dirname, task, model, what)
    if key in _SINGLE:
        return _SINGLE[key]
    import glob
    import os
    from sklearn.metrics import roc_auc_score
    cwd = os.getcwd()
    os.chdir(DEV)
    try:
        from scripts.eval.summarize_u1_clean import clean_matches
        clean = clean_matches()
    finally:
        os.chdir(cwd)
    sc = DEV / "data" / "cache" / dirname / "preds"
    meta = np.load(sc / f"{task}_test_meta.npz", allow_pickle=True)
    ok = np.array([int(m) in clean for m in meta["match_id"]])
    y = meta["y"].astype(int)
    out = {}
    for f in sorted(glob.glob(str(sc / f"{task}_{model}_s*.npz"))):
        seed = int(f.rsplit("_s", 1)[1].split(".")[0])
        z = np.load(f)
        if task == "dest":
            out[seed] = float((z["top1"][ok] == y[ok]).mean()) if what == "metric" else 1 + float(z["logp_true"][ok].mean()) / math.log(96)
        else:
            out[seed] = float(roc_auc_score(y[ok], z["prob"][ok]))
    _SINGLE[key] = out
    return out


def mean_sd(vals):
    v = np.array(list(vals.values()) if isinstance(vals, dict) else vals, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan")), len(v)


# ---------------------------------------------------------------- LaTeX tables

def write_table(name, body_lines, caption, label, env="table", note=None, size="\\footnotesize", colspec=None, tabular="tabular", width=None):
    """Writes a complete booktabs table. body_lines are all lines from the header to the bottom rule (including \\toprule and the like)"""
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    head = f"\\begin{{{tabular}}}{{{width}}}{{{colspec}}}" if tabular == "tabularx" else f"\\begin{{{tabular}}}{{{colspec}}}"
    lines = [f"% Generated by scripts/paper/{name}.py",
             f"\\begin{{{env}}}[tbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
             "\\setlength{\\tabcolsep}{3pt}", head] + body_lines + [f"\\end{{{tabular}}}"]
    if note:
        lines += ["\\par\\smallskip", "\\begin{minipage}{\\linewidth}", "\\scriptsize " + note, "\\end{minipage}"]
    lines += [f"\\end{{{env}}}", ""]
    p = TAB_DIR / f"{name}.tex"
    p.write_text("\n".join(lines), encoding="utf8")
    return p


# ---------------------------------------------------------------- figure style

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", gray="#8a8984", dark="#3b3a37",
         ink="#0b0b0b", ink2="#52514e", grid="#e6e5e1", surface="#ffffff")
CM = 1 / 2.54


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.linewidth": 0.6, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"], "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 1.2,
    })


# colours and line styles of the paper figures
OLD = dict(navy="#1F4E79", red="#CC3311", gray="#888888", lgray="#A0A0A0", stem="#D7DEE8", grid="#E6E6E6",
           edge="#333333", ink="#1A1A1A", ring="#222222", magenta="#882255", band="#EEEEEE", navy_l="#9DB4CC", red_l="#E8A595")


def set_style_old():
    """Style of the paper figures, sized for a printed width of 17.5 cm (body text about 8 pt)"""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.9, "axes.edgecolor": OLD["edge"],
        "axes.labelcolor": OLD["ink"], "xtick.color": OLD["edge"], "ytick.color": OLD["edge"], "text.color": OLD["ink"],
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150, "legend.frameon": False,
        "mathtext.fontset": "dejavusans",
    })


def panel_label(ax, letter, title=None, x=-0.02, y=1.04, fig=None):
    """Panel label (a) in bold at the top left, followed by a bold title"""
    txt = f"({letter})" + (f" {title}" if title else "")
    ax.text(x, y, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold", color=OLD["ink"])


def ringed_scatter(ax, x, y, color, s=52, marker="o", zorder=4):
    """Filled point with a white edge and a thin dark outer ring"""
    ax.scatter(x, y, s=s, marker=marker, facecolor=color, edgecolor="white", linewidth=0.8, zorder=zorder)
    ax.scatter(x, y, s=s * 1.25, marker=marker, facecolor="none", edgecolor=OLD["ring"], linewidth=0.55, zorder=zorder)


VLABELS = []


def vlabel(ax, x, y, value, nd, **kw):
    """Draws a number label on a figure and records (displayed text, value); save_fig writes them to results/figure_data/{name}_labels.csv"""
    txt = f"{value:.{nd}f}".replace("-", "\u2212")
    ax.text(x, y, txt, **kw)
    VLABELS.append((txt.replace("\u2212", "-"), float(value)))
    return txt


def save_fig(fig, name):
    if VLABELS:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / f"{name}_labels.csv", "w", encoding="utf8") as fh:
            fh.write("text,value\n" + "".join(f"{t},{v!r}\n" for t, v in VLABELS))
        VLABELS.clear()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches=None, facecolor="white")
    return FIG_DIR / f"{name}.pdf"
