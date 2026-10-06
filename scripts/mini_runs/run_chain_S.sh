#!/bin/bash
# 队列 S  拼接融合加落点空间输入打乱，三个种子，和已有的门控版 shuf_dest 配对，看门控是否把一个任务的噪声隔在该任务之内
# 等队列 P 结束后启动。只用 CPU，线程数 3
# 输入  同 run_chain_unified.sh   输出  data/cache/u1_clean/u1concat_shuf_dest_s*.* 和 logs/mini_chain_S_2026-10-02.log
# 运行  nohup bash scripts/run_chain_S.sh > logs/mini_chain_S_2026-10-02.log 2>&1 &
# Last modified 2026-10-06 (public release: repository-relative cd, no conda activation; queue logic unchanged since 2026-10-02)
cd "$(dirname "$0")/../.." || exit 1   # repository root
# activate the Python environment of requirements.txt before launching this script
export TASG_THREADS=3 PYTHONPATH=. PYTHONUNBUFFERED=1
echo "[S] $(date +%H:%M:%S) 等队列 P 结束"
until grep -q "ALL DONE" logs/mini_chain_P_2026-10-02.log; do sleep 20; done
N=0; T0=$(date +%s)
run() {
  tag=$1; seed=$2; shift 2; N=$((N+1))
  echo "[S $N] $(date +%H:%M:%S) start $tag seed=$seed  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
  caffeinate -i python -u scripts/training/train_unified_clean.py --seed "$seed" --tag "$tag" "$@"
  echo "[S $N] $(date +%H:%M:%S) end $tag seed=$seed rc=$?  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
}
run u1concat_shuf_dest 0 --fusion concat --shuffle_spatial dest
run u1concat_shuf_dest 1 --fusion concat --shuffle_spatial dest
run u1concat_shuf_dest 2 --fusion concat --shuffle_spatial dest
echo "[S] ALL DONE $(date +%H:%M:%S)"
