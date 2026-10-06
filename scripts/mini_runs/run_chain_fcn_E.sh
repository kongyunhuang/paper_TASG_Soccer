#!/bin/bash
# 保留分辨率编码器的追加实验，队列 E  立即启动（占第二个进程位），只用 CPU，线程数 3
# 输入  同 run_chain_unified.sh   输出  data/cache/u1_clean/ 和 logs/mini_chain_fcn_E_2026-10-02.log
# 运行  nohup bash scripts/run_chain_fcn_E.sh > logs/mini_chain_fcn_E_2026-10-02.log 2>&1 &
# Last modified 2026-10-06 (public release: repository-relative cd, no conda activation; queue logic unchanged since 2026-10-02)
cd "$(dirname "$0")/../.." || exit 1   # repository root
# activate the Python environment of requirements.txt before launching this script
export TASG_THREADS=3 PYTHONPATH=. PYTHONUNBUFFERED=1
N=0; TOTAL=2; T0=$(date +%s)
run() {
  tag=$1; seed=$2; shift 2; N=$((N+1))
  echo "[fcnE $N/$TOTAL] $(date +%H:%M:%S) start $tag seed=$seed  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
  caffeinate -i python -u scripts/training/train_unified_clean.py --seed "$seed" --tag "$tag" "$@"
  echo "[fcnE $N/$TOTAL] $(date +%H:%M:%S) end $tag seed=$seed rc=$?  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
}
run u1fcn 1 --encoder fcn
run u1fcn_shuf_dest 0 --encoder fcn --shuffle_spatial dest
echo "[fcnE] ALL DONE $(date +%H:%M:%S)"
