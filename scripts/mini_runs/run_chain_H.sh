#!/bin/bash
# 队列 H  Tackle 干预补第二、第三个种子。等队列 E（u1fcn_shuf_dest s0）跑完腾出进程位后启动，只用 CPU，线程数 3
# 输入  同 run_chain_unified.sh   输出  data/cache/u1_clean/shuf_tackle_s1.* s2.* 和 logs/mini_chain_H_2026-10-02.log
# 运行  nohup bash scripts/run_chain_H.sh > logs/mini_chain_H_2026-10-02.log 2>&1 &
# Last modified 2026-10-06 (public release: repository-relative cd, no conda activation; queue logic unchanged since 2026-10-02)
cd "$(dirname "$0")/../.." || exit 1   # repository root
# activate the Python environment of requirements.txt before launching this script
export TASG_THREADS=3 PYTHONPATH=. PYTHONUNBUFFERED=1
echo "[H] $(date +%H:%M:%S) 等队列 E 结束"
until grep -q "ALL DONE" logs/mini_chain_fcn_E_2026-10-02.log; do sleep 30; done
N=0; TOTAL=2; T0=$(date +%s)
run() {
  tag=$1; seed=$2; shift 2; N=$((N+1))
  echo "[H $N/$TOTAL] $(date +%H:%M:%S) start $tag seed=$seed  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
  caffeinate -i python -u scripts/training/train_unified_clean.py --seed "$seed" --tag "$tag" "$@"
  echo "[H $N/$TOTAL] $(date +%H:%M:%S) end $tag seed=$seed rc=$?  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
}
run shuf_tackle 1 --shuffle_spatial tackle
run shuf_tackle 2 --shuffle_spatial tackle
echo "[H] ALL DONE $(date +%H:%M:%S)"
