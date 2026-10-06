#!/usr/bin/env python3
"""
论文第二版图表的公共工具  读权威结果文件、记录每个数的出处、写 LaTeX 表、统一作图样式
============================
所有 tab_*.py 和 fig_*.py 都从这里读数。每个进表进图的数都登记一条出处（文件、行号和列名，或 json 字段，
或由哪些已登记的数算出来），登记表写到 scripts/paper_v2/out/{名字}_sources.json，verify_all.py 按它逐条回读核对。
  md 表格      按行号取一行，用该表的表头把单元格对上列名，单元格里的“均值 ± 标准差 n=种子数”可再拆成几部分
  json         按字段路径取值
  逐事件文件   24/25 的逐种子数从 data/cache 的逐事件预测重算，口径照搬 scripts/eval/summarize_u1_clean.py（直接导入它的函数）
本文件只读结果文件，不写 data/、audit/、eval/、training/ 下的任何东西。

输入  audit/*.md，data/cache/holdout_2526/*.json，data/cache/u1_clean/*.npz，data/cache/single_clean{,_rich}/preds/*.npz
输出  无（被其他脚本导入）

Usage
  conda activate kronos
  PYTHONPATH=. python -c "import scripts.paper_v2.common"   (from the repository root)

Last modified 2026-10-06 (public release: paths relative to the repository root, outputs/ instead of 02_manuscript/)
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

DEV = Path(__file__).resolve().parents[2]          # repository root (01_dev_workspace in the development tree)
ROOT = DEV                                          # provenance entries are written relative to this
MANU = ROOT / "outputs"                             # generated tables and figures
TAB_DIR = MANU / "tables"
FIG_DIR = MANU / "figures" / "v2"
OUT_DIR = DEV / "scripts" / "paper_v2" / "out"

SRC = {
    "u1": DEV / "audit" / "u1_clean_summary.md",
    "rules": DEV / "audit" / "dest_rule_baselines.md",
    "pref": DEV / "audit" / "player_decision_preference_summary.md",
    "hold_md": DEV / "audit" / "holdout_2526_summary.md",
    "hold_json": DEV / "data" / "cache" / "holdout_2526" / "summary.json",
    "hold_rules": DEV / "data" / "cache" / "holdout_2526" / "dest_rules_holdout.json",
    "train_log": DEV / "logs" / "u1_clean_u1_s0_2026-10-02.log",
    "frozen": DEV / "memory_logs" / "2026-10-02_留出数据确认_冻结方案和判定标准.md",
    "crit_2ch": DEV / "memory_logs" / "2026-10-02_两通道统一模型_预先写下的判定标准.md",
    "crit_rich": DEV / "memory_logs" / "2026-10-02_加厚标量统一模型_预先写下的判定标准.md",
    "crit_concat": DEV / "memory_logs" / "2026-10-02_拼接版落点干预对照_预先写下的判定标准.md",
    "crit_pen": DEV / "memory_logs" / "2026-10-02_门控代价实验_预先写下的判定标准.md",
    "crit_hard": DEV / "memory_logs" / "2026-10-02_硬门控实验_预先写下的判定标准.md",
    "crit_enc": DEV / "memory_logs" / "2026-10-02_主模型编码器_预先写下的选择规则.md",
    "crit_pref": DEV / "memory_logs" / "2026-10-02_球员决策偏好_预先写下的判定标准.md",
}

TASKS = ["pass", "dest", "shot", "interception", "ball_recovery", "pressure", "dribble", "tackle"]
OUTCOME = [t for t in TASKS if t != "dest"]
TASK_NAME = {"pass": "Pass", "dest": "Destination", "shot": "Shot", "interception": "Interception",
             "ball_recovery": "Ball recovery", "pressure": "Pressure", "dribble": "Dribble", "tackle": "Tackle"}


def rel(p: Path) -> str:
    """相对 xCV2 根目录的路径，出处里统一这样写"""
    return str(Path(p).resolve().relative_to(ROOT))


def md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- markdown 表格

def _lines(path):
    return Path(path).read_text(encoding="utf8").split("\n")


def _split(line):
    parts = line.strip().split("|")
    return [c.strip() for c in parts[1:-1]]


def md_row(path, line):
    """取第 line 行（从 1 数）所在 markdown 表的这一行，返回 {列名: 单元格字符串}"""
    L = _lines(path)
    i = line - 1
    assert L[i].strip().startswith("|"), f"{rel(path)} L{line} 不是表格行  {L[i][:60]}"
    j = i
    while j > 0 and not re.match(r"^\|\s*:?-{3}", L[j].strip()):
        j -= 1
    header = _split(L[j - 1])
    cells = _split(L[i])
    assert len(header) == len(cells), f"{rel(path)} L{line} 列数和表头不一致"
    return dict(zip(header, cells))


def md_find(path, heading_prefix, nth_table=0, **keys):
    """在以 heading_prefix 开头的标题之后的第 nth_table 张表里，找各列等于 keys 的那一行，返回行号（从 1 数）"""
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
                raise KeyError(f"{rel(path)} {heading_prefix} 表 {nth_table} 里没有 {keys}")
        i += 1
    raise KeyError(f"{rel(path)} 找不到 {heading_prefix} 后的第 {nth_table} 张表")


def parse_cell(s):
    """把单元格拆成几部分。'0.2710 ± 0.0004（0.0947） n=5' -> mean sd paren n；纯数字 -> value；方括号区间 -> lo hi"""
    s = s.strip().replace("−", "-")
    out = {}
    m = re.search(r"n=(\d+)", s)
    if m:
        out["n"] = int(m.group(1))
    m = re.search(r"（([-+0-9.]+)）", s)
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
        raise KeyError(f"{rel(path)} L{line} 列 {col} 单元格 {cell!r} 没有 {part}")
    return p[part]


def json_value(path, keys):
    d = json.load(open(path, encoding="utf8"))
    for k in keys:
        d = d[k]
    return d


# ---------------------------------------------------------------- 出处登记

class Registry:
    """每个进表进图的数登记一条。src 是可回读的定位，display 是表里或图里实际出现的字符串"""

    def __init__(self, name):
        self.name = name
        self.items = []
        self.notes = []

    def add(self, iid, value, src, display=None, cell=None, fmt=None):
        it = dict(id=iid, value=None if value is None else float(value) if isinstance(value, (int, float, np.floating, np.integer)) else value,
                  src=src, display=display, cell=cell, fmt=fmt)
        self.items.append(it)
        return value

    # 常用的几种出处
    def md(self, iid, path, line, col, part="value", fmt=None, cell=None):
        v = md_value(path, line, col, part)
        d = fmt_num(v, fmt) if fmt else None
        self.add(iid, v, dict(kind="md", file=rel(path), line=line, col=col, part=part), display=d, cell=cell, fmt=fmt)
        return v

    def mdre(self, iid, path, line, col, regex, group=1, fmt=None, cell=None, sign=1):
        """单元格里用正则取数。col 为 None 时对整行取。sign=-1 用于“负 0.0042”这类写法"""
        s = md_row(path, line)[col] if col else _lines(path)[line - 1]
        m = re.search(regex, s.replace("−", "-"))
        assert m, f"{rel(path)} L{line} 列 {col} 用 {regex!r} 取不到  {s[:80]}"
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
        assert must_contain in L[line - 1], f"{rel(path)} L{line} 不含 {must_contain!r}"
        self.add(iid, text, dict(kind="text", file=rel(path), line=line, must_contain=must_contain), display=text, cell=cell)
        return text

    def save(self, extra=None):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        srcs = sorted({it["src"]["file"] for it in self.items if "file" in it["src"]})
        d = dict(name=self.name, items=self.items, notes=self.notes,
                 source_md5={f: md5(ROOT / f) for f in srcs})
        if extra:
            d.update({k: (rel(v) if k == "data_csv" else v) for k, v in extra.items()})
        (OUT_DIR / f"{self.name}_sources.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf8")


def fmt_num(v, fmt):
    """fmt 取 'f4'（四位小数）、'f3'、'f5'、'pct1'（百分数一位小数）、'int'（千分位整数）、'sf4'（带正号四位小数）、'sf5'"""
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
    """把显示字符串转成 LaTeX，负号用数学减号"""
    return re.sub(r"(?<![0-9A-Za-z])-(?=[0-9.])", "$-$", s).replace("+0.", "$+$0.").replace("±", "$\\pm$")


def norm_tex(s):
    """verify 用  把 LaTeX 单元格还原成纯文本，便于和 display 比"""
    s = s.replace("$-$", "-").replace("$+$", "+").replace("$\\pm$", "±").replace("$\\times$", "×")
    s = re.sub(r"\\makecell(\[[a-z]\])?", "", s)
    s = re.sub(r"\\(scriptsize|footnotesize|small|textsc|textit|textbf|emph)", "", s)
    s = s.replace("\\\\[-2pt]", " ").replace("\\\\", " ").replace("{", "").replace("}", "").replace("$", "").replace("\\,", "").replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- 24/25 逐种子数（从逐事件文件重算）

_RUNS = None


def unified_runs():
    """用 summarize_u1_clean.load_runs 在 202 场上重算全部统一模型运行，返回 {(tag, seed): run}"""
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
    """what 取 'metric'（落点 top-1，其余 AUC）、'skill'、'gate'。返回 {种子: 值}"""
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
    """同数据单任务模型在 202 场上的逐种子值，口径照搬 summarize_u1_clean.single_task_table。
    what 取 'metric'（AUC，落点是 top-1）或 'skill'（落点 1 减对数损失除以 ln96）。返回 {种子: 值}"""
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


# ---------------------------------------------------------------- LaTeX 表

def write_table(name, body_lines, caption, label, env="table", note=None, size="\\footnotesize", colspec=None, tabular="tabular", width=None):
    """写一张完整的 booktabs 表。body_lines 是表头到底线之间的所有行（含 \\toprule 等）"""
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    head = f"\\begin{{{tabular}}}{{{width}}}{{{colspec}}}" if tabular == "tabularx" else f"\\begin{{{tabular}}}{{{colspec}}}"
    lines = [f"% 由 scripts/paper_v2/{name}.py 生成，不要手改；数据出处见 outputs/tables/数据来源_2026-10-04.md",
             f"\\begin{{{env}}}[tbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
             "\\setlength{\\tabcolsep}{3pt}", head] + body_lines + [f"\\end{{{tabular}}}"]
    if note:
        lines += ["\\par\\smallskip", "\\begin{minipage}{\\linewidth}", "\\scriptsize " + note, "\\end{minipage}"]
    lines += [f"\\end{{{env}}}", ""]
    p = TAB_DIR / f"{name}.tex"
    p.write_text("\n".join(lines), encoding="utf8")
    return p


# ---------------------------------------------------------------- 作图样式

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


# 5 月旧图（scripts/plotting/plot_main_heatmap.py、plot_gate_vs_delta_auc.py）的配色和样式，10-04 用户要求四张数据图照它重做
OLD = dict(navy="#1F4E79", red="#CC3311", gray="#888888", lgray="#A0A0A0", stem="#D7DEE8", grid="#E6E6E6",
           edge="#333333", ink="#1A1A1A", ring="#222222", magenta="#882255", band="#EEEEEE", navy_l="#9DB4CC", red_l="#E8A595")


def set_style_old():
    """旧图样式按 17.5 cm 的实际宽度换算  旧图 8.2 到 11 英寸宽、正文字 9.5 到 12 pt，缩到 17.5 cm 后约 8 pt"""
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
    """面板标签 (a) 加粗放左上角，后接加粗标题"""
    txt = f"({letter})" + (f" {title}" if title else "")
    ax.text(x, y, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold", color=OLD["ink"])


def ringed_scatter(ax, x, y, color, s=52, marker="o", zorder=4):
    """旧图的点  实心带白边，外面再套一圈深色细环"""
    ax.scatter(x, y, s=s, marker=marker, facecolor=color, edgecolor="white", linewidth=0.8, zorder=zorder)
    ax.scatter(x, y, s=s * 1.25, marker=marker, facecolor="none", edgecolor=OLD["ring"], linewidth=0.55, zorder=zorder)


VLABELS = []


def vlabel(ax, x, y, value, nd, **kw):
    """图上标的数值，格式化后画出来，同时记下（显示文字，原值），save_fig 时写到 out/{图名}_labels.csv 供 verify_all 核对"""
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
