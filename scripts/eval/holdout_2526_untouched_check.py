#!/usr/bin/env python3
"""
留出数据确认，第一步  核实 25/26 的 146 场今天没有被任何模型评估过
============================
只读。取 L1_events_v3.parquet 里 season_dir 含 2025_26 的全部 match_id（应为 146 场），然后
  一  扫 data/cache/ 下所有结果文件（npz、parquet、json、jsonl，不含 99_archive），凡带 match_id 一类字段的，
      数它和这 146 场的交集，逐文件列出。空间张量缓存 action_soccermaps_*_idx.parquet 是输入不是结果，单独列出并注明。
  二  扫 scripts/ 下所有 .py，列出每个读 L1_events_v3.parquet 的脚本以及它按什么赛季过滤（FULL_SEASONS、TRAIN_SEASONS、
      TEST_SEASONS、显式字符串），凡是没有过滤或显式包含 2025_26 的，单独标出。
  三  任何“同一个键出现两次”的情况直接报错，不静默覆盖。
输出到终端和 audit/holdout_2526_untouched_check.md。不写 data/cache/，不碰原始文件。

输入  data/cache/L1_events_v3.parquet，data/cache/**，scripts/**/*.py
输出  audit/holdout_2526_untouched_check.md

Usage
  conda activate kronos
  PYTHONPATH=. python -u scripts/eval/holdout_2526_untouched_check.py

Last modified 2026-10-02
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CACHE = Path("data/cache")
OUT = Path("audit/holdout_2526_untouched_check.md")
INPUT_INDEX_PATTERN = re.compile(r"action_soccermaps_.*_idx\.parquet$|perspective_fixed/soccermaps_.*_idx\.parquet$")
MATCH_KEY = re.compile(r"match", re.I)


def holdout_ids():
    df = pd.read_parquet(CACHE / "L1_events_v3.parquet", columns=["match_id", "season_dir", "event_id"])
    s = df[df["season_dir"].str.contains("2025_26")]
    ids = set(s["match_id"].astype(int).unique())
    eids = set(s["event_id"].astype(str))
    per = s.groupby("season_dir")["match_id"].nunique().to_dict()
    return ids, eids, per


def scan_cache_event_ids(eids):
    """按 event_id 再扫一遍。空间张量索引只有 event_id 没有 match_id，单任务模型的预测也只存 event_id，按 match 扫不到。"""
    rows = {}
    files = sorted(p for p in glob.glob(str(CACHE / "**" / "*"), recursive=True) if os.path.isfile(p))
    for f in files:
        rel = os.path.relpath(f, CACHE)
        if rel in rows:
            raise RuntimeError(f"同一个文件键出现两次  {rel}")
        ext = f.rsplit(".", 1)[-1].lower() if "." in os.path.basename(f) else ""
        kind = "输入索引" if INPUT_INDEX_PATTERN.search(rel) else "结果或中间文件"
        try:
            if ext == "npz":
                z = np.load(f, allow_pickle=True)
                keys = [k for k in z.files if re.search(r"event|eid", k, re.I)]
                for k in keys:
                    vals = set(np.asarray(z[k]).astype(str).ravel().tolist())
                    rows[rel] = dict(kind=kind, fields=k, hit=len(vals & eids), n=len(vals))
            elif ext == "parquet":
                sch = pq.read_schema(f)
                cols = [c for c in sch.names if re.search(r"event_id|eid", c, re.I)]
                if not cols:
                    continue
                t = pq.read_table(f, columns=cols).to_pandas()
                hits = {c: len(set(t[c].astype(str)) & eids) for c in cols}
                rows[rel] = dict(kind=kind, fields=",".join(cols), hit=max(hits.values()), n=int(max(t[c].nunique() for c in cols)))
        except Exception as e:  # noqa: BLE001
            rows[rel] = dict(kind=kind, fields=f"读取失败 {type(e).__name__}", hit=None, n=None)
    return rows


def ints_in(obj, out, depth=0):
    """递归收集 json 里所有整数（含整数字符串），只用于和 146 个 match_id 求交集。"""
    if depth > 12:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            ints_in(v, out, depth + 1)
            if isinstance(k, str) and k.isdigit():
                out.add(int(k))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            ints_in(v, out, depth + 1)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, np.integer)):
        out.add(int(obj))
    elif isinstance(obj, str) and obj.isdigit():
        out.add(int(obj))


def scan_cache(ids):
    rows = {}
    files = sorted(p for p in glob.glob(str(CACHE / "**" / "*"), recursive=True) if os.path.isfile(p))
    for f in files:
        rel = os.path.relpath(f, CACHE)
        if rel in rows:
            raise RuntimeError(f"同一个文件键出现两次  {rel}")
        ext = f.rsplit(".", 1)[-1].lower() if "." in os.path.basename(f) else ""
        kind = "输入索引" if INPUT_INDEX_PATTERN.search(rel) else "结果或中间文件"
        try:
            if ext == "npz":
                z = np.load(f, allow_pickle=True)
                keys = [k for k in z.files if MATCH_KEY.search(k)]
                if not keys:
                    rows[rel] = dict(kind=kind, fields="无 match 字段", hit=0, n=0)
                    continue
                for k in keys:
                    v = z[k]
                    try:
                        vals = set(np.asarray(v).astype(np.int64).ravel().tolist())
                    except (ValueError, TypeError):
                        rows[rel] = dict(kind=kind, fields=f"{k} 非整数", hit=0, n=len(np.asarray(v).ravel()))
                        continue
                    rows[rel] = dict(kind=kind, fields=k, hit=len(vals & ids), n=len(vals))
            elif ext == "parquet":
                sch = pq.read_schema(f)
                cols = [c for c in sch.names if MATCH_KEY.search(c)]
                if not cols:
                    rows[rel] = dict(kind=kind, fields="无 match 列", hit=0, n=0)
                    continue
                t = pq.read_table(f, columns=cols).to_pandas()
                hits = {c: len(set(pd.to_numeric(t[c], errors="coerce").dropna().astype(np.int64)) & ids) for c in cols}
                nn = {c: int(t[c].nunique()) for c in cols}
                rows[rel] = dict(kind=kind, fields=",".join(cols), hit=max(hits.values()), n=max(nn.values()))
            elif ext in ("json", "jsonl"):
                found = set()
                with open(f) as fh:
                    if ext == "json":
                        ints_in(json.load(fh), found)
                    else:
                        for line in fh:
                            line = line.strip()
                            if line:
                                ints_in(json.loads(line), found)
                rows[rel] = dict(kind=kind, fields="json 内所有整数", hit=len(found & ids), n=len(found))
            else:
                rows[rel] = dict(kind=kind, fields=f"未扫描（{ext or '无扩展名'}）", hit=None, n=None)
        except Exception as e:  # noqa: BLE001
            rows[rel] = dict(kind=kind, fields=f"读取失败 {type(e).__name__}", hit=None, n=None)
    return rows


def scan_scripts():
    rows = []
    for f in sorted(glob.glob("scripts/**/*.py", recursive=True)):
        src = open(f, encoding="utf8", errors="replace").read()
        if "L1_events_v3" not in src:
            continue
        flags = []
        for name in ("FULL_SEASONS", "TRAIN_SEASONS", "TEST_SEASONS", "build_dataset", "2025_26", "318", "season_dir"):
            if name in src:
                flags.append(name)
        imports_build = bool(re.search(r"from scripts\.training\.train_unified_clean import[^\n]*build_dataset", src))
        rows.append(dict(script=f, flags=" ".join(flags), imports_build_dataset=imports_build,
                         mentions_2526=("2025_26" in src) or ("_318_" in src)))
    return rows


def main():
    t0 = time.time()
    ids, eids, per = holdout_ids()
    print(f"[ids] 25/26 比赛 {len(ids)} 场  事件 {len(eids):,}  {per}", flush=True)
    rows = scan_cache(ids)
    print(f"[cache] 按 match_id 扫了 {len(rows)} 个文件  ({time.time() - t0:.0f}s)", flush=True)
    erows = scan_cache_event_ids(eids)
    print(f"[cache] 按 event_id 扫了 {len(erows)} 个带 event 字段的文件  ({time.time() - t0:.0f}s)", flush=True)
    hits = {k: v for k, v in rows.items() if v["hit"]}
    res_hits = {k: v for k, v in hits.items() if v["kind"] != "输入索引"}
    ehits = {k: v for k, v in erows.items() if v["hit"]}
    eres_hits = {k: v for k, v in ehits.items() if v["kind"] != "输入索引"}
    scripts = scan_scripts()
    L = ["# 留出数据确认，25/26 是否被今天的任何模型评估过（只读检查）", "",
         f"由 `scripts/eval/holdout_2526_untouched_check.py` 生成，{time.strftime('%Y-%m-%d %H:%M')}。",
         f"25/26 比赛 {len(ids)} 场（{per}），事件 {len(eids):,} 个。", "",
         "## 零  按 event_id 扫（空间张量索引和单任务预测只有 event_id）", "",
         f"带 event 字段的文件 {len(erows)} 个，有交集的 {len(ehits)} 个，其中结果或中间文件 **{len(eres_hits)}** 个（L1_events_v3.parquet 本身是输入，列在这里只为完整）。", "",
         "| 文件 | 类别 | 字段 | 命中事件数 | 文件内事件数 |", "|---|---|---|---:|---:|"]
    L += [f"| {k} | {v['kind']} | {v['fields']} | {v['hit']:,} | {v['n']:,} |" for k, v in sorted(ehits.items())]
    L += ["", "带 event 字段但命中为 0 的文件", "", "| 文件 | 字段 | 文件内事件数 |", "|---|---|---:|"]
    L += [f"| {k} | {v['fields']} | {v['n']:,} |" for k, v in sorted(erows.items()) if v["hit"] == 0]
    L += ["", "## 一  data/cache/ 下带 match 字段的文件和 146 场的交集", "",
          f"扫了 {len(rows)} 个文件。有交集的 {len(hits)} 个，其中结果或中间文件 **{len(res_hits)}** 个，输入索引 {len(hits) - len(res_hits)} 个。", ""]
    if res_hits:
        L += ["**结果或中间文件里命中 25/26 的（必须逐个解释）**", "", "| 文件 | 字段 | 命中场次 | 文件内场次数 |", "|---|---|---:|---:|"]
        L += [f"| {k} | {v['fields']} | {v['hit']} | {v['n']} |" for k, v in sorted(res_hits.items())]
        L.append("")
    L += ["输入索引里含 25/26 的（空间张量缓存本来就覆盖 25/26，不是模型成绩）", "", "| 文件 | 字段 | 命中场次 | 文件内场次数 |", "|---|---|---:|---:|"]
    L += [f"| {k} | {v['fields']} | {v['hit']} | {v['n']} |" for k, v in sorted(hits.items()) if v["kind"] == "输入索引"]
    L += ["", "带 match 字段但命中为 0 的文件", "", "| 文件 | 字段 | 文件内场次数 |", "|---|---|---:|"]
    L += [f"| {k} | {v['fields']} | {v['n']} |" for k, v in sorted(rows.items()) if v["hit"] == 0 and v["n"]]
    skipped = [k for k, v in rows.items() if v["hit"] is None]
    L += ["", f"没扫的文件（二进制张量、权重、日志等，共 {len(skipped)} 个）", ""]
    L += [f"- {k}  {rows[k]['fields']}" for k in skipped]
    L += ["", "## 二  scripts/ 下读 L1_events_v3.parquet 的脚本及其赛季过滤", "",
          "| 脚本 | 出现的过滤标识 | 导入 build_dataset | 文本里提到 25/26 |", "|---|---|---|---|"]
    L += [f"| {r['script']} | {r['flags']} | {r['imports_build_dataset']} | {r['mentions_2526']} |" for r in scripts]
    L += ["", "读法。导入 build_dataset 的脚本一律先按 FULL_SEASONS 过滤，拿不到 25/26。其余脚本要逐个看它的过滤逻辑（见方案文件里的逐个说明）。", ""]
    OUT.write_text("\n".join(L), encoding="utf8")
    print(f"[hits] 结果或中间文件命中 {len(res_hits)} 个", flush=True)
    for k, v in sorted(hits.items()):
        print(f"   {v['kind']:8s} {k:70s} {v['fields']:30s} hit {v['hit']:4d} / {v['n']}")
    print(f"[scripts] 读 L1 的脚本 {len(scripts)} 个，文本里提到 25/26 的 {sum(r['mentions_2526'] for r in scripts)} 个")
    for r in scripts:
        print(f"   {r['script']:60s} {r['flags']:60s} build={r['imports_build_dataset']} 2526={r['mentions_2526']}")
    print(f"[done] {OUT}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
