#!/bin/bash
# 统一模型实验链，在 Mac mini 上顺序跑，只用 CPU，线程数 3
# 输入  仓库根目录下的缓存数据   输出  data/cache/u1_clean/ 和 logs/mini_chain_unified_2026-10-02.log
# 运行  nohup bash scripts/run_chain_unified.sh > logs/mini_chain_unified_2026-10-02.log 2>&1 &
# Last modified 2026-10-06 (public release: repository-relative cd, no conda activation; queue logic unchanged since 2026-10-02)
cd "$(dirname "$0")/../.." || exit 1   # repository root
# activate the Python environment of requirements.txt before launching this script
export TASG_THREADS=3 PYTHONPATH=. PYTHONUNBUFFERED=1
N=0; TOTAL=10; T0=$(date +%s)
run() {
  tag=$1; seed=$2; shift 2; N=$((N+1))
  echo "[chain $N/$TOTAL] $(date +%H:%M:%S) start $tag seed=$seed  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
  caffeinate -i python -u scripts/training/train_unified_clean.py --seed "$seed" --tag "$tag" "$@"
  echo "[chain $N/$TOTAL] $(date +%H:%M:%S) end $tag seed=$seed rc=$?  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
}
run shuf_dest 2 --shuffle_spatial dest
run u1concat 0 --fusion concat
run u1concat 1 --fusion concat
run u1concat 2 --fusion concat
run u1fcn 0 --encoder fcn
run u1_2ch 0 --channels 2
run u1_2ch 1 --channels 2
run u1_2ch 2 --channels 2
run shuf_pressure 0 --shuffle_spatial pressure
run shuf_tackle 0 --shuffle_spatial tackle
echo "[chain] ALL DONE $(date +%H:%M:%S)"
