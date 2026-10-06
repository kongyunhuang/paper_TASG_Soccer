#!/bin/bash
# 队列 T2  队列 T 的第二条进程位，同一组七个运行按倒序接，和队列 T 靠 u1_clean/*.running 占位文件互不重复（已有结果的秒跳）。
#          23:25 在 Mac mini 本机起的，当时同传已停、机器只剩队列 T 一条在跑，用户要求提速。只用 CPU，线程数 3，和其他运行口径一致
# 输入  同 run_chain_unified.sh   输出  data/cache/u1_clean/ 和 logs/mini_chain_T2_2026-10-02.log
# 运行  nohup bash scripts/run_chain_T2.sh > logs/mini_chain_T2_2026-10-02.log 2>&1 &
# Last modified 2026-10-06 (public release: repository-relative cd, no conda activation; queue logic unchanged since 2026-10-02)
cd "$(dirname "$0")/../.." || exit 1   # repository root
# activate the Python environment of requirements.txt before launching this script
export TASG_THREADS=3 PYTHONPATH=. PYTHONUNBUFFERED=1
N=0; T0=$(date +%s)
run() {
  tag=$1; seed=$2; shift 2; N=$((N+1))
  echo "[T2 $N] $(date +%H:%M:%S) start $tag seed=$seed  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
  caffeinate -i python -u scripts/training/train_unified_clean.py --seed "$seed" --tag "$tag" "$@"
  echo "[T2 $N] $(date +%H:%M:%S) end $tag seed=$seed rc=$?  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
}
run u1pen010 3 --gate_penalty 0.01
run u1pen010 2 --gate_penalty 0.01
run u1pen010 1 --gate_penalty 0.01
run u1richpen010 3 --rich_scalars --gate_penalty 0.01
run u1richpen010 2 --rich_scalars --gate_penalty 0.01
run u1richpen010 1 --rich_scalars --gate_penalty 0.01
run u1rich 3 --rich_scalars
echo "[T2] ALL DONE $(date +%H:%M:%S)"
