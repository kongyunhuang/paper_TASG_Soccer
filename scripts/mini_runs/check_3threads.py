"""
三线程对照  在 Mac mini 上用 3 线程重跑 dribble seed0 的三个神经网络模型，和 MacBook 已有结果逐位比对
输入  data/cache/single_clean/dribble.jsonl (MacBook 的结果)   输出  只打印，写到临时目录
Usage  TASG_THREADS=3 PYTHONPATH=. python -u check_3threads.py
Last modified 2026-10-02 (unchanged for the public release)
"""
import json, sys, tempfile, pathlib
import scripts.training.train_single_clean as m
ref = {(r["model"], r["seed"]): r for r in map(json.loads, open(m.OUT_DIR / "dribble.jsonl"))}
tmp = pathlib.Path(tempfile.mkdtemp())
m.OUT_DIR = tmp; m.PRED_DIR = tmp / "preds"
sys.argv = ["x", "--tasks", "dribble", "--seeds", "0", "--models", "M2_MLP_360,M4_CNN_Full,G1_Gating"]
m.main()
ok = True
for line in open(tmp / "dribble.jsonl"):
    r = json.loads(line); q = ref[(r["model"], r["seed"])]
    same = r["test"]["auc"] == q["test"]["auc"] and r["val"]["auc"] == q["val"]["auc"]
    ok &= same
    print(f"[check] {r['model']} mini test={r['test']['auc']:.10f} val={r['val']['auc']:.10f} | macbook test={q['test']['auc']:.10f} val={q['val']['auc']:.10f} | {'SAME' if same else 'DIFF'} host_ref={q.get('host')}", flush=True)
print("[check] RESULT", "ALL_SAME" if ok else "MISMATCH", flush=True)
