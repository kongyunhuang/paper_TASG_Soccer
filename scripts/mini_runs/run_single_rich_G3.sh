#!/bin/bash
# 队列 G3  单任务神经网络模型（M2, M4, G1）加厚标量版，种子 0 到 2。等队列 S 跑完腾出进程位后启动，线程数 3
# 输入  同 train_single_clean.py   输出  data/cache/single_clean_rich/ 和 logs/mini_single_rich_2026-10-02.log
# 运行  nohup bash scripts/run_single_rich_G3.sh >> logs/mini_single_rich_2026-10-02.log 2>&1 &
# Last modified 2026-10-06 (public release: repository-relative cd, no conda activation; queue logic unchanged since 2026-10-02)
cd "$(dirname "$0")/../.." || exit 1   # repository root
# activate the Python environment of requirements.txt before launching this script
export TASG_THREADS=3 PYTHONPATH=. PYTHONUNBUFFERED=1
echo "[G3] $(date +%H:%M:%S) 等队列 S 结束"
until grep -q "ALL DONE" logs/mini_chain_S_2026-10-02.log; do sleep 60; done
echo "[G3] $(date +%H:%M:%S) 开始"
caffeinate -i python -u scripts/training/train_single_clean.py --rich_scalars --models M2_MLP_360,M4_CNN_Full,G1_Gating --seeds 0,1,2
echo "[G3] ALL DONE $(date +%H:%M:%S) rc=$?"
