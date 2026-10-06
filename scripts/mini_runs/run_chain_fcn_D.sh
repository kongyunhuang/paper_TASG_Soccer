#!/bin/bash
# 保留分辨率编码器的追加实验，队列 D  等跨机器对照（u1chk）跑完后启动，只用 CPU，线程数 3
# 输入  同 run_chain_unified.sh   输出  data/cache/u1_clean/ 和 logs/mini_chain_fcn_D_2026-10-02.log
# 运行  nohup bash scripts/run_chain_fcn_D.sh > logs/mini_chain_fcn_D_2026-10-02.log 2>&1 &
# Last modified 2026-10-06 (public release: repository-relative cd, no conda activation; queue logic unchanged since 2026-10-02)
cd "$(dirname "$0")/../.." || exit 1   # repository root
# activate the Python environment of requirements.txt before launching this script
export TASG_THREADS=3 PYTHONPATH=. PYTHONUNBUFFERED=1
echo "[fcnD] $(date +%H:%M:%S) 等 u1chk 结束"
until grep -q "ALL DONE" logs/mini_u1_check_2026-10-02.log; do sleep 60; done
N=0; TOTAL=2; T0=$(date +%s)
run() {
  tag=$1; seed=$2; shift 2; N=$((N+1))
  echo "[fcnD $N/$TOTAL] $(date +%H:%M:%S) start $tag seed=$seed  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
  caffeinate -i python -u scripts/training/train_unified_clean.py --seed "$seed" --tag "$tag" "$@"
  echo "[fcnD $N/$TOTAL] $(date +%H:%M:%S) end $tag seed=$seed rc=$?  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
}
run u1fcn 2 --encoder fcn
run u1fcnconcat 0 --encoder fcn --fusion concat
echo "[fcnD] ALL DONE $(date +%H:%M:%S)"
