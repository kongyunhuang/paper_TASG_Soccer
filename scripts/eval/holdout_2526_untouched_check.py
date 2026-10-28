#!/usr/bin/env python3
"""
holdout_2526_untouched_check.py, check that no model has been evaluated on the 25/26 hold-out matches
====================================================================================================
Read only. Takes every match_id of L1_events_v3.parquet whose season_dir contains 2025_26 (146 matches expected), then
  1  scans every result file under data/cache/ (npz, parquet, json, jsonl). For each file with a match_id-like field, counts
     its overlap with the 146 matches, file by file. The spatial tensor indices action_soccermaps_*_idx.parquet are inputs,
     not results, and are listed separately.
  2  scans every .py under scripts/ and lists each script that reads L1_events_v3.parquet together with its season filter
     (FULL_SEASONS, TRAIN_SEASONS, TEST_SEASONS, explicit strings); scripts without a filter or that explicitly include
     2025_26 are flagged.
  3  any duplicate key raises an error instead of being silently overwritten.
Output to the terminal and to results/holdout_2526_untouched_check.md. Nothing is written to data/cache/, and the raw files
are not touched.

Inputs   data/cache/L1_events_v3.parquet, data/cache/**, scripts/**/*.py
Outputs  results/holdout_2526_untouched_check.md

Usage
  PYTHONPATH=. python -u scripts/eval/holdout_2526_untouched_check.py

Last modified 2026-10-28
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
OUT = Path("results/holdout_2526_untouched_check.md")
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
    """Second scan by event_id. The spatial tensor indices hold only event_id, not match_id, and so do the single-task predictions, so a scan by match misses them."""
    rows = {}
    files = sorted(p for p in glob.glob(str(CACHE / "**" / "*"), recursive=True) if os.path.isfile(p))
    for f in files:
        rel = os.path.relpath(f, CACHE)
        if rel in rows:
            raise RuntimeError(f"duplicate file key  {rel}")
        ext = f.rsplit(".", 1)[-1].lower() if "." in os.path.basename(f) else ""
        kind = "input index" if INPUT_INDEX_PATTERN.search(rel) else "result or intermediate"
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
            rows[rel] = dict(kind=kind, fields=f"read failed {type(e).__name__}", hit=None, n=None)
    return rows


def ints_in(obj, out, depth=0):
    """Recursively collect every integer in a json (integer strings included), used only for the overlap with the 146 match_ids."""
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
            raise RuntimeError(f"duplicate file key  {rel}")
        ext = f.rsplit(".", 1)[-1].lower() if "." in os.path.basename(f) else ""
        kind = "input index" if INPUT_INDEX_PATTERN.search(rel) else "result or intermediate"
        try:
            if ext == "npz":
                z = np.load(f, allow_pickle=True)
                keys = [k for k in z.files if MATCH_KEY.search(k)]
                if not keys:
                    rows[rel] = dict(kind=kind, fields="no match field", hit=0, n=0)
                    continue
                for k in keys:
                    v = z[k]
                    try:
                        vals = set(np.asarray(v).astype(np.int64).ravel().tolist())
                    except (ValueError, TypeError):
                        rows[rel] = dict(kind=kind, fields=f"{k} not integer", hit=0, n=len(np.asarray(v).ravel()))
                        continue
                    rows[rel] = dict(kind=kind, fields=k, hit=len(vals & ids), n=len(vals))
            elif ext == "parquet":
                sch = pq.read_schema(f)
                cols = [c for c in sch.names if MATCH_KEY.search(c)]
                if not cols:
                    rows[rel] = dict(kind=kind, fields="no match column", hit=0, n=0)
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
                rows[rel] = dict(kind=kind, fields="all integers in json", hit=len(found & ids), n=len(found))
            else:
                rows[rel] = dict(kind=kind, fields=f"not scanned ({ext or 'no extension'})", hit=None, n=None)
        except Exception as e:  # noqa: BLE001
            rows[rel] = dict(kind=kind, fields=f"read failed {type(e).__name__}", hit=None, n=None)
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
    print(f"[ids] 25/26 matches {len(ids)}  events {len(eids):,}  {per}", flush=True)
    rows = scan_cache(ids)
    print(f"[cache] scanned {len(rows)} files by match_id  ({time.time() - t0:.0f}s)", flush=True)
    erows = scan_cache_event_ids(eids)
    print(f"[cache] scanned {len(erows)} files with an event field by event_id  ({time.time() - t0:.0f}s)", flush=True)
    hits = {k: v for k, v in rows.items() if v["hit"]}
    res_hits = {k: v for k, v in hits.items() if v["kind"] != "input index"}
    ehits = {k: v for k, v in erows.items() if v["hit"]}
    eres_hits = {k: v for k, v in ehits.items() if v["kind"] != "input index"}
    scripts = scan_scripts()
    L = ["# Hold-out confirmation, whether any model has been evaluated on 25/26 (read-only check)", "",
         f"Generated by `scripts/eval/holdout_2526_untouched_check.py`, {time.strftime('%Y-%m-%d %H:%M')}.",
         f"25/26 matches {len(ids)} ({per}), events {len(eids):,}.", "",
         "## 0  Scan by event_id (the spatial tensor indices and the single-task predictions hold only event_id)", "",
         f"Files with an event field {len(erows)}, with overlap {len(ehits)}, of which result or intermediate files **{len(eres_hits)}** (L1_events_v3.parquet itself is an input and is listed only for completeness).", "",
         "| File | Kind | Fields | Events hit | Events in file |", "|---|---|---|---:|---:|"]
    L += [f"| {k} | {v['kind']} | {v['fields']} | {v['hit']:,} | {v['n']:,} |" for k, v in sorted(ehits.items())]
    L += ["", "Files with an event field and no hit", "", "| File | Fields | Events in file |", "|---|---|---:|"]
    L += [f"| {k} | {v['fields']} | {v['n']:,} |" for k, v in sorted(erows.items()) if v["hit"] == 0]
    L += ["", "## 1  Files under data/cache/ with a match field, and their overlap with the 146 matches", "",
          f"{len(rows)} files scanned. {len(hits)} overlap, of which **{len(res_hits)}** result or intermediate files and {len(hits) - len(res_hits)} input indices.", ""]
    if res_hits:
        L += ["**Result or intermediate files that hit 25/26 (each must be explained)**", "", "| File | Fields | Matches hit | Matches in file |", "|---|---|---:|---:|"]
        L += [f"| {k} | {v['fields']} | {v['hit']} | {v['n']} |" for k, v in sorted(res_hits.items())]
        L.append("")
    L += ["Input indices that contain 25/26 (the spatial tensor cache covers 25/26 by design; these are not model results)", "", "| File | Fields | Matches hit | Matches in file |", "|---|---|---:|---:|"]
    L += [f"| {k} | {v['fields']} | {v['hit']} | {v['n']} |" for k, v in sorted(hits.items()) if v["kind"] == "input index"]
    L += ["", "Files with a match field and no hit", "", "| File | Fields | Matches in file |", "|---|---|---:|"]
    L += [f"| {k} | {v['fields']} | {v['n']} |" for k, v in sorted(rows.items()) if v["hit"] == 0 and v["n"]]
    skipped = [k for k, v in rows.items() if v["hit"] is None]
    L += ["", f"Files not scanned (binary tensors, weights, logs and others, {len(skipped)} in total)", ""]
    L += [f"- {k}  {rows[k]['fields']}" for k in skipped]
    L += ["", "## 2  Scripts under scripts/ that read L1_events_v3.parquet, and their season filters", "",
          "| Script | Filter markers present | Imports build_dataset | Mentions 25/26 in the text |", "|---|---|---|---|"]
    L += [f"| {r['script']} | {r['flags']} | {r['imports_build_dataset']} | {r['mentions_2526']} |" for r in scripts]
    L += ["", "How to read. Scripts that import build_dataset filter by FULL_SEASONS first and cannot reach 25/26. The filters of the other scripts have to be read script by script.", ""]
    OUT.write_text("\n".join(L), encoding="utf8")
    print(f"[hits] result or intermediate files hit {len(res_hits)}", flush=True)
    for k, v in sorted(hits.items()):
        print(f"   {v['kind']:8s} {k:70s} {v['fields']:30s} hit {v['hit']:4d} / {v['n']}")
    print(f"[scripts] scripts reading L1 {len(scripts)}, mentioning 25/26 in the text {sum(r['mentions_2526'] for r in scripts)}")
    for r in scripts:
        print(f"   {r['script']:60s} {r['flags']:60s} build={r['imports_build_dataset']} 2526={r['mentions_2526']}")
    print(f"[done] {OUT}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
