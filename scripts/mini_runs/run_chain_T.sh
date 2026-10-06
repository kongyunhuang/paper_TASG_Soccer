#!/bin/bash
# 队列 T  软门控加代价的确认步（种子 1、2、3，λ 0.01，标量加厚和薄各三个）和它需要的基准 u1rich s3。等队列 Q2 全部跑完后启动。只用 CPU，线程数 3
# 输入  同 run_chain_unified.sh   输出  data/cache/u1_clean/ 和 logs/mini_chain_T_2026-10-02.log
# 运行  nohup bash scripts/run_chain_T.sh > logs/mini_chain_T_2026-10-02.log 2>&1 &
# Last modified 2026-10-06 (public release: repository-relative cd, no conda activation; queue logic unchanged since 2026-10-02)
cd "$(dirname "$0")/../.." || exit 1   # repository root
# activate the Python environment of requirements.txt before launching this script
export TASG_THREADS=3 PYTHONPATH=. PYTHONUNBUFFERED=1
echo "[T] $(date +%H:%M:%S) 等队列 Q2 结束"
until grep -q "ALL DONE" logs/mini_chain_Q2_2026-10-02.log; do sleep 60; done
N=0; T0=$(date +%s)
run() {
  tag=$1; seed=$2; shift 2; N=$((N+1))
  echo "[T $N] $(date +%H:%M:%S) start $tag seed=$seed  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
  caffeinate -i python -u scripts/training/train_unified_clean.py --seed "$seed" --tag "$tag" "$@"
  echo "[T $N] $(date +%H:%M:%S) end $tag seed=$seed rc=$?  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
}
run u1rich 3 --rich_scalars
run u1richpen010 1 --rich_scalars --gate_penalty 0.01
run u1richpen010 2 --rich_scalars --gate_penalty 0.01
run u1richpen010 3 --rich_scalars --gate_penalty 0.01
run u1pen010 1 --gate_penalty 0.01
run u1pen010 2 --gate_penalty 0.01
run u1pen010 3 --gate_penalty 0.01
echo "[T] ALL DONE $(date +%H:%M:%S)"
