#!/bin/bash
# 队列 R  球员决策偏好的正式检验（应用案例），等到 20:05（上课用同传结束）后启动，先跑种子 0 的两折，再跑种子 1、2。只用 CPU，线程数 4
# 输入  同 scripts/eval/player_decision_preference.py   输出  data/cache/player_pref/ 和 logs/mini_app_R_2026-10-02.log
# 运行  nohup bash scripts/run_app_R.sh > logs/mini_app_R_2026-10-02.log 2>&1 &
# Last modified 2026-10-06 (public release: repository-relative cd, no conda activation; queue logic unchanged since 2026-10-02)
cd "$(dirname "$0")/../.." || exit 1   # repository root
# activate the Python environment of requirements.txt before launching this script
export TASG_THREADS=4 PYTHONPATH=. PYTHONUNBUFFERED=1
echo "[R] $(date +%H:%M:%S) 等到 20:05 再启动"
until [ "$(date +%H%M)" -ge 2005 ]; do sleep 60; done
echo "[R] $(date +%H:%M:%S) 开始种子 0"
caffeinate -i python -u scripts/eval/player_decision_preference.py --folds 1,2 --seeds 0
echo "[R] $(date +%H:%M:%S) 种子 0 结束 rc=$?，开始种子 1、2"
caffeinate -i python -u scripts/eval/player_decision_preference.py --folds 1,2 --seeds 1,2
echo "[R] ALL DONE $(date +%H:%M:%S) rc=$?"
