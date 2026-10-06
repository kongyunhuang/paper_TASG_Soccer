#!/bin/zsh
# 一键重做论文第二版的全部图表并逐数核对
# ============================
# 依次跑七张表、四张正文数据图、三张补充图的生成脚本，最后跑 verify_all.py 回读每个数、写数据来源 md。任何一步出错就停。
# fig_overview 10-04 起停用（稿子换回作者手画的 figures/fig1.pdf），不再重出，它的出处登记仍在 out/ 里供核对。
# 每步打印开始时间和耗时，整套约一两分钟。
#
# 输入  audit/、data/cache/、logs/、memory_logs/ 下的结果文件（只读）
# 输出  outputs/tables/tab_*.tex，outputs/figures/v2/fig_*.{pdf,png}，outputs/tables/数据来源_2026-10-04.md，
#       scripts/paper_v2/out/
#
# Usage
#   zsh scripts/paper_v2/make_all.sh        (from the repository root, inside the Python environment of requirements.txt)
#
# Last modified 2026-10-06 (public release: no conda activation inside the script; paths relative to the repository root; pipefail added so a failing step stops the chain)
set -e
set -o pipefail   # a failing python step must stop the chain even though its output is piped through grep
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
for s in tab_tasks tab_outcome_main tab_destination tab_interventions tab_holdout tab_gate_variants_supp tab_application_supp \
         fig_outcome_vs_scalars fig_destination fig_gate fig_holdout \
         figS1_gate_vs_shuffle figS2_soft_penalty figS3_player_preference verify_all; do
  t0=$(date +%s)
  echo "[make_all] $(date +%H:%M:%S) 开始 $s"
  python -u scripts/paper_v2/$s.py 2>&1 | grep -v -E "NOT subset|post.stringData|timestamp seems very low"
  echo "[make_all] $s 用时 $(( $(date +%s) - t0 )) 秒"
done
echo "[make_all] 全部完成"
