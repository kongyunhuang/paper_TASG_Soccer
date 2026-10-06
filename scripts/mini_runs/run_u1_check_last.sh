#!/bin/bash
# 统一模型的跨机器对照  等两条队列都跑完后，在 Mac mini 上重跑 u1 seed0（标签 u1chk），和 MacBook 的 u1_s0 逐位比对
# 输入  同 run_chain_unified.sh   输出  data/cache/u1_clean/u1chk_s0.* 和 logs/mini_u1_check_2026-10-02.log
# 运行  nohup bash scripts/run_u1_check_last.sh > logs/mini_u1_check_2026-10-02.log 2>&1 &
# Last modified 2026-10-06 (public release: repository-relative cd, no conda activation; queue logic unchanged since 2026-10-02)
cd "$(dirname "$0")/../.." || exit 1   # repository root
# activate the Python environment of requirements.txt before launching this script
export TASG_THREADS=3 PYTHONPATH=. PYTHONUNBUFFERED=1
echo "[u1chk] $(date +%H:%M:%S) 等两条队列结束"
until grep -q "ALL DONE" logs/mini_chain_unified_2026-10-02.log && grep -q "ALL DONE" logs/mini_chain_unified_B_2026-10-02.log; do sleep 60; done
echo "[u1chk] $(date +%H:%M:%S) 开始"
caffeinate -i python -u scripts/training/train_unified_clean.py --seed 0 --tag u1chk
echo "[u1chk] ALL DONE $(date +%H:%M:%S) rc=$?"
