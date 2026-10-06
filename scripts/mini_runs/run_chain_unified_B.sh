#!/bin/bash
# 统一模型第二条队列  等单任务那条跑完腾出 3 个线程后启动，从队列后半段往前接，和第一条队列靠占位文件互不重复
# 输入  同 run_chain_unified.sh   输出  data/cache/u1_clean/ 和 logs/mini_chain_unified_B_2026-10-02.log
# 运行  nohup bash scripts/run_chain_unified_B.sh > logs/mini_chain_unified_B_2026-10-02.log 2>&1 &
# Last modified 2026-10-06 (public release: repository-relative cd, no conda activation; queue logic unchanged since 2026-10-02)
cd "$(dirname "$0")/../.." || exit 1   # repository root
# activate the Python environment of requirements.txt before launching this script
export TASG_THREADS=3 PYTHONPATH=. PYTHONUNBUFFERED=1
echo "[chainB] $(date +%H:%M:%S) 等单任务进程结束"
while pgrep -f "train_single_clean.py" >/dev/null; do sleep 60; done
echo "[chainB] $(date +%H:%M:%S) 单任务进程已结束，开始"
N=0; TOTAL=9; T0=$(date +%s)
run() {
  tag=$1; seed=$2; shift 2; N=$((N+1))
  echo "[chainB $N/$TOTAL] $(date +%H:%M:%S) start $tag seed=$seed  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
  caffeinate -i python -u scripts/training/train_unified_clean.py --seed "$seed" --tag "$tag" "$@"
  echo "[chainB $N/$TOTAL] $(date +%H:%M:%S) end $tag seed=$seed rc=$?  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
}
run u1_2ch 0 --channels 2
run u1_2ch 1 --channels 2
run u1_2ch 2 --channels 2
run shuf_pressure 0 --shuffle_spatial pressure
run shuf_tackle 0 --shuffle_spatial tackle
run u1fcn 0 --encoder fcn
run u1concat 2 --fusion concat
run u1concat 1 --fusion concat
run u1concat 0 --fusion concat
echo "[chainB] ALL DONE $(date +%H:%M:%S)"
