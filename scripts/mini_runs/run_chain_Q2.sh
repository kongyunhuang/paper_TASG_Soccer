#!/bin/bash
# 队列 Q2  硬门控加代价的探索步（种子 0，标量加厚和薄各两个 λ），之后补 Tackle 干预的两个种子和保留分辨率编码器第三个种子
# 等队列 E（u1fcn_shuf_dest s0）跑完腾出进程位后启动。只用 CPU，线程数 3
# 输入  同 run_chain_unified.sh   输出  data/cache/u1_clean/ 和 logs/mini_chain_Q2_2026-10-02.log
# 运行  nohup bash scripts/run_chain_Q2.sh > logs/mini_chain_Q2_2026-10-02.log 2>&1 &
# Last modified 2026-10-06 (public release: repository-relative cd, no conda activation; queue logic unchanged since 2026-10-02)
cd "$(dirname "$0")/../.." || exit 1   # repository root
# activate the Python environment of requirements.txt before launching this script
export TASG_THREADS=3 PYTHONPATH=. PYTHONUNBUFFERED=1
echo "[Q2] $(date +%H:%M:%S) 等队列 E 结束"
until grep -q "ALL DONE" logs/mini_chain_fcn_E_2026-10-02.log; do sleep 30; done
N=0; T0=$(date +%s)
run() {
  tag=$1; seed=$2; shift 2; N=$((N+1))
  echo "[Q2 $N] $(date +%H:%M:%S) start $tag seed=$seed  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
  caffeinate -i python -u scripts/training/train_unified_clean.py --seed "$seed" --tag "$tag" "$@"
  echo "[Q2 $N] $(date +%H:%M:%S) end $tag seed=$seed rc=$?  (已用 $(( ($(date +%s)-T0)/60 )) 分钟)"
}
run u1richhard010 0 --rich_scalars --hard_gate --gate_penalty 0.01
run u1richhard030 0 --rich_scalars --hard_gate --gate_penalty 0.03
run u1hard010 0 --hard_gate --gate_penalty 0.01
run u1hard030 0 --hard_gate --gate_penalty 0.03
run shuf_tackle 1 --shuffle_spatial tackle
run shuf_tackle 2 --shuffle_spatial tackle
run u1fcn 2 --encoder fcn
echo "[Q2] ALL DONE $(date +%H:%M:%S)"
