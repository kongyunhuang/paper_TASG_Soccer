#!/bin/bash
# 队列 U  留出数据确认第一阶段要的重训（只碰 24/25），两条并行。U1 重训统一模型 u1 种子 1、2 并核对；U2 单任务薄标量五个种子、加厚三个种子。线程数各 4
# 输入  同各脚本文件头   输出  data/cache/holdout_2526/{retrain,single,single_rich}/ 和 logs/mini_holdout_*.log
# 运行  nohup bash scripts/run_holdout_U.sh U1 > logs/mini_holdout_U1_2026-10-03.log 2>&1 &   以及 U2
# Last modified 2026-10-06 (public release: repository-relative cd, no conda activation; queue logic unchanged since 2026-10-03)
cd "$(dirname "$0")/../.." || exit 1   # repository root
# activate the Python environment of requirements.txt before launching this script
export TASG_THREADS=4 PYTHONPATH=. PYTHONUNBUFFERED=1
Q=$1; T0=$(date +%s)
step() { echo "[$Q] $(date +%H:%M:%S) start $*  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"; caffeinate -i python -u "$@"; echo "[$Q] $(date +%H:%M:%S) end rc=$? $*  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"; }
if [ "$Q" = "U1" ]; then
  step scripts/eval/holdout_2526_retrain_unified.py --seed 1 --verify
  step scripts/eval/holdout_2526_retrain_unified.py --seed 2 --verify
else
  step scripts/eval/holdout_2526_single_task.py --seeds 0,1,2,3,4
  step scripts/eval/holdout_2526_single_task.py --rich_scalars --seeds 0,1,2
fi
echo "[$Q] ALL DONE $(date +%H:%M:%S)"
