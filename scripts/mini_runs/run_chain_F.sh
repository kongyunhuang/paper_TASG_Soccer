#!/bin/bash
# 队列 F  等跨机器对照（u1chk）跑完后启动。先跑加厚标量的统一模型三个种子，再跑保留分辨率编码器第三个种子和它的拼接对照
# 输入  同 run_chain_unified.sh   输出  data/cache/u1_clean/ 和 logs/mini_chain_F_2026-10-02.log
# 运行  nohup bash scripts/run_chain_F.sh > logs/mini_chain_F_2026-10-02.log 2>&1 &
# Last modified 2026-10-06 (public release: repository-relative cd, no conda activation; queue logic unchanged since 2026-10-02)
cd "$(dirname "$0")/../.." || exit 1   # repository root
# activate the Python environment of requirements.txt before launching this script
export TASG_THREADS=3 PYTHONPATH=. PYTHONUNBUFFERED=1
echo "[F] $(date +%H:%M:%S) 等 u1chk 结束"
until grep -q "ALL DONE" logs/mini_u1_check_2026-10-02.log; do sleep 30; done
N=0; TOTAL=5; T0=$(date +%s)
run() {
  tag=$1; seed=$2; shift 2; N=$((N+1))
  echo "[F $N/$TOTAL] $(date +%H:%M:%S) start $tag seed=$seed  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
  caffeinate -i python -u scripts/training/train_unified_clean.py --seed "$seed" --tag "$tag" "$@"
  echo "[F $N/$TOTAL] $(date +%H:%M:%S) end $tag seed=$seed rc=$?  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
}
run u1rich 0 --rich_scalars
run u1rich 1 --rich_scalars
run u1rich 2 --rich_scalars
run u1fcn 2 --encoder fcn
run u1fcnconcat 0 --encoder fcn --fusion concat
echo "[F] ALL DONE $(date +%H:%M:%S)"
