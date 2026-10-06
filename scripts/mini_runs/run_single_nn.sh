#!/bin/bash
# 单任务神经网络模型 (M2, M4, G1) 种子 1 到 4，在 Mac mini 上跑，只用 CPU，线程数 3，已完成的条目自动跳过
# 输入  同 train_single_clean.py   输出  data/cache/single_clean/ 和 logs/mini_single_nn_s1to4_2026-10-02.log
# 运行  nohup bash scripts/run_single_nn.sh > logs/mini_single_nn_s1to4_2026-10-02.log 2>&1 &
# Last modified 2026-10-06 (public release: repository-relative cd, no conda activation; queue logic unchanged since 2026-10-02)
cd "$(dirname "$0")/../.." || exit 1   # repository root
# activate the Python environment of requirements.txt before launching this script
export TASG_THREADS=3 PYTHONPATH=. PYTHONUNBUFFERED=1
caffeinate -i python -u scripts/training/train_single_clean.py --models M2_MLP_360,M4_CNN_Full,G1_Gating --seeds 1,2,3,4
echo "[single] ALL DONE $(date +%H:%M:%S) rc=$?"
